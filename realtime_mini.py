import os
import json
import time
import uuid
import base64
import asyncio
import logging

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import websockets
from dotenv import load_dotenv

# ============================================================
#  Load env + config
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", "5025"))  # Different port for testing
VOICE = os.getenv("VOICE", "sage")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Either put your full system prompt in the env var...
VOICE_SYSTEM_PROMPT = os.getenv("VOICE_SYSTEM_PROMPT")

# ...or paste it directly here as a fallback:
if not VOICE_SYSTEM_PROMPT:
    VOICE_SYSTEM_PROMPT = """
You are the Zapstrix Voice Agent.

answer any questions the user asks to the best of your abilities.
"""


"""
- You answer calls from small business owners and their customers.
- You speak clearly, concisely, and sound like a professional human assistant.
- You never mention that you are using OpenAI, Twilio, or any underlying tools.
- You ask clarifying questions when needed and confirm details like dates, times, and names.
- If the caller is rambling, gently steer the conversation back to the task.
- Always be polite, calm, and efficient."""

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")

# ============================================================
#  Logging
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LOG_EVENT_TYPES = [
    "error",
    "session.created",
    "session.updated",
    "response.done",
    "rate_limits.updated",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
]

# Track active sessions (mainly for debugging / health)
ACTIVE_SESSIONS = {}

# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI(title="Zapstrix Voice Agent (Mini)", version="3.0.1-mini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  Basic health / index
# ============================================================


@app.get("/")
async def index():
    return {
        "status": "operational",
        "service": "Zapstrix Voice Agent (Mini)",
        "version": "3.0.1-mini",
        "model": "gpt-4o-mini-realtime-preview",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "version": "3.0.1-mini",
        "model": "gpt-4o-mini-realtime-preview",
    }


# ============================================================
#  Twilio inbound: /incoming-call → TwiML with Media Stream
# ============================================================


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """
    Twilio hits this when your Twilio number is called.
    We respond with TwiML that opens a Media Stream to /media-stream.
    """
    try:
        host = request.headers.get("host") or request.url.netloc
        stream_url = f"wss://{host}/media-stream"

        logger.info(f"Incoming call - Media Stream URL: {stream_url}")

        from twilio.twiml.voice_response import VoiceResponse, Connect

        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=stream_url)
        response.append(connect)

        return HTMLResponse(content=str(response), media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in /incoming-call: {e}")
        from twilio.twiml.voice_response import VoiceResponse

        response = VoiceResponse()
        response.say(
            "We're sorry, but we're experiencing technical difficulties. Please try again later."
        )
        return HTMLResponse(content=str(response), media_type="application/xml")


# ============================================================
#  WebSocket bridge: Twilio <-> OpenAI Realtime
# ============================================================


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio opens this WebSocket and sends JSON events:
    - "start"
    - "media" (audio chunks, base64 G.711 mu-law)
    - "stop"
    We proxy to OpenAI Realtime and send its audio back.
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()
    logger.info(f"Twilio WebSocket accepted - Session: {session_id}")

    ACTIVE_SESSIONS[session_id] = {
        "start_time": time.time(),
        "stream_sid": None,
    }

    try:
        logger.info("Connecting to OpenAI Realtime API (MINI MODEL)...")
        async with websockets.connect(
            # Using gpt-4o-mini-realtime-preview for 60% cost savings
            f"wss://api.openai.com/v1/realtime?"
            f"model=gpt-4o-mini-realtime-preview&temperature={TEMPERATURE}",
            extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            ping_interval=30,
            ping_timeout=10,
        ) as openai_ws:
            logger.info("Connected to OpenAI Realtime API (gpt-4o-mini-realtime-preview)")

            # Configure the Realtime session (correct, latest schema)
            await initialize_session(openai_ws)

            # Bridge audio both ways
            await asyncio.gather(
                _twilio_to_openai(websocket, openai_ws, session_id),
                _openai_to_twilio(websocket, openai_ws, session_id),
            )

    except Exception as e:
        logger.error(f"Error in /media-stream handler: {e}")
    finally:
        ACTIVE_SESSIONS.pop(session_id, None)
        logger.info(f"Session ended: {session_id}")


# ============================================================
#  OpenAI session init
# ============================================================


async def send_initial_conversation_item(openai_ws):
    """
    Have the AI talk first by seeding a 'user' message
    and triggering response.create.
    """
    initial_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the caller according to your system instructions and ask how you can help.",
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(initial_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """
    Configure the Realtime session using the latest API shape.
    We use audio/pcmu (G.711 μ-law) to match Twilio Media Streams.
    """
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-4o-mini-realtime-preview",  # Mini model
            # Only need audio back from the model; text is optional
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    # Twilio sends G.711 μ-law as base64-encoded bytes
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE,
                },
            },
            # IMPORTANT: this must be the FULL text of your system prompt,
            # not a pmpt_ ID.
            "instructions": VOICE_SYSTEM_PROMPT,
        },
    }

    logger.info("Sending session.update (instructions length: %d chars)",
                len(VOICE_SYSTEM_PROMPT or ""))

    await openai_ws.send(json.dumps(session_update))

    # Make the AI speak first
    await send_initial_conversation_item(openai_ws)


# ============================================================
#  Twilio -> OpenAI
# ============================================================


async def _twilio_to_openai(
    twilio_ws: WebSocket, openai_ws: websockets.WebSocketClientProtocol, session_id: str
):
    """
    Receive Twilio Media Stream events and forward audio
    to OpenAI Realtime as input_audio_buffer.append.
    """
    try:
        async for message in twilio_ws.iter_text():
            data = json.loads(message)
            event_type = data.get("event")

            if event_type == "start":
                stream_sid = data["start"]["streamSid"]
                ACTIVE_SESSIONS[session_id]["stream_sid"] = stream_sid
                logger.info(f"Twilio stream started - SID: {stream_sid}")

            elif event_type == "media":
                # Twilio sends base64-encoded G.711 μ-law audio in data['media']['payload']
                if openai_ws.open:
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": data["media"]["payload"],
                    }
                    await openai_ws.send(json.dumps(audio_append))

            elif event_type == "stop":
                logger.info("Twilio stream stopped")
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in Twilio -> OpenAI loop: {e}")


# ============================================================
#  OpenAI -> Twilio
# ============================================================


async def _openai_to_twilio(
    twilio_ws: WebSocket, openai_ws: websockets.WebSocketClientProtocol, session_id: str
):
    """
    Receive Realtime events from OpenAI and send audio back
    to Twilio as media events.
    """
    stream_sid = None
    first_audio = True
    message_count = 0

    try:
        async for raw in openai_ws:
            message_count += 1
            response = json.loads(raw)
            event_type = response.get("type")

            if event_type in LOG_EVENT_TYPES:
                logger.debug(f"OpenAI event #{message_count}: {event_type} {response}")

            # Grab stream SID once we have it from Twilio
            if stream_sid is None:
                stream_sid = ACTIVE_SESSIONS.get(session_id, {}).get("stream_sid")

            # Audio packets from OpenAI → Twilio media events
            if event_type == "response.output_audio.delta" and response.get("delta"):
                if not stream_sid:
                    # We don't yet know where to send this audio
                    continue

                try:
                    # The delta is base64-encoded audio/pcmu.
                    # Re-encode to be safe before sending to Twilio.
                    decoded = base64.b64decode(response["delta"])
                    audio_payload = base64.b64encode(decoded).decode("utf-8")

                    if first_audio:
                        logger.info(
                            f"✅ First audio packet from OpenAI - bytes: {len(decoded)}"
                        )
                        first_audio = False

                    audio_delta = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_payload},
                    }
                    await twilio_ws.send_json(audio_delta)

                except Exception as e:
                    logger.error(f"Error processing OpenAI audio delta: {e}")

            # Transcript of AI's response (for logs/debugging)
            elif event_type == "response.output_audio_transcript.delta":
                # incremental transcript chunks; optional
                pass

            elif event_type == "response.output_audio_transcript.done":
                transcript = response.get("transcript") or ""
                if transcript:
                    logger.info(f"AI said: {transcript}")

            # User speech transcript
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript") or ""
                if transcript:
                    logger.info(f"User said: {transcript}")

            # Errors
            elif event_type == "error":
                logger.error(f"OpenAI error event: {response}")

    except Exception as e:
        logger.error(f"Error in OpenAI -> Twilio loop: {e}")
    finally:
        logger.info(f"OpenAI stream ended - messages processed: {message_count}")


# ============================================================
#  Main entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Zapstrix Voice Agent (MINI) on port {PORT}")
    logger.info("Using model: gpt-4o-mini-realtime-preview (60% cost savings)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
