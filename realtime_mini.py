import os
import json
import time
import uuid
import base64
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import websockets
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ============================================================
#  Load env + config
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", "5025"))  # Different port for testing
VOICE = os.getenv("VOICE", "sage")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Google Sheets configuration
GOOGLE_SHEETS_CREDS_JSON = os.getenv("GOOGLE_SHEETS_CREDS_JSON")  # JSON string of service account credentials
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Voice Agent Call Logs")

# Either put your full system prompt in the env var...
VOICE_SYSTEM_PROMPT = os.getenv("VOICE_SYSTEM_PROMPT")

# ...or paste it directly here as a fallback:
if not VOICE_SYSTEM_PROMPT:
    VOICE_SYSTEM_PROMPT = """
You are Gabriel Pascual's personal voice assistant.

Gabriel is a Data/AI specialist based in Tampa, FL. He builds AI solutions and full-stack applications. He's also a photographer.

YOUR GOAL: Find out why they're calling and help schedule a meeting with Gabriel.

IMPORTANT RULES:
- Ask ONE question at a time, then STOP and wait for their answer
- Listen carefully - let them finish speaking before responding
- Keep responses very brief (1-2 sentences maximum)
- If they pause, wait - don't fill silence with more questions
- Be warm but efficient

CONVERSATION FLOW:
1. Greet them and ask how you can help
2. Wait for their response
3. Ask for their name if needed
4. Offer to schedule a meeting
5. Get their contact info and preferred times

Never mention technical tools you're using.
"""

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

# Google Sheets client (initialized lazily)
gsheets_client = None
call_log_sheet = None

# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI(title="Gabriel Pascual's Voice Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  Google Sheets Setup
# ============================================================


def init_google_sheets():
    """
    Initialize Google Sheets client and get the call log sheet.
    Returns the worksheet object or None if initialization fails.
    """
    global gsheets_client, call_log_sheet

    if call_log_sheet is not None:
        return call_log_sheet

    if not GOOGLE_SHEETS_CREDS_JSON:
        logger.warning("GOOGLE_SHEETS_CREDS_JSON not set - call logging to Google Sheets disabled")
        return None

    try:
        # Parse credentials from environment variable (JSON string)
        creds_dict = json.loads(GOOGLE_SHEETS_CREDS_JSON)

        # Set up credentials
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gsheets_client = gspread.authorize(creds)

        # Open the spreadsheet (must be created manually and shared with service account)
        try:
            spreadsheet = gsheets_client.open(GOOGLE_SHEET_NAME)
            logger.info(f"Opened existing Google Sheet: {GOOGLE_SHEET_NAME}")
        except gspread.SpreadsheetNotFound:
            logger.error(f"Google Sheet '{GOOGLE_SHEET_NAME}' not found!")
            logger.error("Please create the sheet manually and share it with the service account email.")
            logger.error(f"Service account email: {creds_dict.get('client_email', 'N/A')}")
            return None

        # Get or create the worksheet
        try:
            call_log_sheet = spreadsheet.worksheet("Call Logs")
        except gspread.WorksheetNotFound:
            call_log_sheet = spreadsheet.add_worksheet(title="Call Logs", rows="1000", cols="8")
            # Add headers
            call_log_sheet.append_row([
                "Timestamp",
                "Call SID",
                "From Number",
                "To Number",
                "Event Type",
                "Duration (sec)",
                "Exchanges",
                "Summary"
            ])
            logger.info("Created Call Logs worksheet with headers")

        logger.info("Google Sheets integration initialized successfully")
        return call_log_sheet

    except json.JSONDecodeError as e:
        logger.error(f"Invalid GOOGLE_SHEETS_CREDS_JSON format: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing Google Sheets: {e}")
        if "storage quota" in str(e).lower() or "403" in str(e):
            logger.error("⚠️ Google Drive storage quota exceeded. Please free up space or use a different account.")
            logger.error("📝 Call logging will fall back to console logs only.")
        return None


# ============================================================
#  Call Logging Utilities
# ============================================================


def log_call_event(call_sid: str, from_number: str, to_number: str, event_type: str, data: dict = None):
    """
    Log call events to Google Sheets for easy access and monitoring.
    Falls back to console logging if Google Sheets is not configured.
    """
    try:
        sheet = init_google_sheets()

        # Prepare row data
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        duration = ""
        exchanges = ""
        summary = ""

        if data:
            duration = data.get("duration", "")
            exchanges = data.get("transcript_count", "")
            summary = data.get("summary", "")

        if sheet is not None:
            row = [
                timestamp,
                call_sid,
                from_number,
                to_number,
                event_type,
                str(duration),
                str(exchanges),
                summary
            ]

            # Append to sheet
            sheet.append_row(row)
            logger.debug(f"Logged call event to Google Sheets: {event_type} for {call_sid}")
        else:
            # Fallback: log to console
            logger.info(f"Call Log: {timestamp} | {event_type} | From: {from_number} | To: {to_number} | Summary: {summary}")

    except Exception as e:
        logger.error(f"Error logging call event to Google Sheets: {e}")
        # Fallback to console
        logger.info(f"Call Log (fallback): {event_type} | From: {from_number} | CallSID: {call_sid}")


def generate_call_summary(transcripts: list) -> str:
    """
    Generate a narrative summary from the conversation transcripts.
    """
    if not transcripts:
        return "No conversation recorded"

    user_messages = [t for t in transcripts if t.get("role") == "user"]
    ai_messages = [t for t in transcripts if t.get("role") == "assistant"]

    # Build a conversational summary
    summary_parts = []

    # Who called and what was discussed
    if user_messages:
        user_text = " ".join([m.get("content", "") for m in user_messages])

        # Create a brief narrative summary
        if len(user_text) > 10:
            summary_parts.append(f"Caller said: {user_text[:200]}...")

        # Add AI responses that contain key info (like scheduling, next steps)
        key_phrases = ["schedule", "meeting", "call back", "email", "contact", "available", "time"]
        ai_summary = []
        for msg in ai_messages:
            content = msg.get("content", "").lower()
            if any(phrase in content for phrase in key_phrases):
                ai_summary.append(msg.get("content", ""))

        if ai_summary:
            summary_parts.append(f"Response: {' '.join(ai_summary)[:200]}...")

    if not summary_parts:
        return f"Brief call - {len(transcripts)} exchanges"

    return " | ".join(summary_parts)


# ============================================================
#  Basic health / index
# ============================================================


@app.get("/")
async def index():
    return {
        "status": "operational",
        "service": "Gabriel Pascual's Voice Agent",
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
        # Parse form data from Twilio
        form_data = await request.form()
        caller_number = form_data.get("From", "Unknown")
        called_number = form_data.get("To", "Unknown")
        call_sid = form_data.get("CallSid", "Unknown")

        # Log incoming call
        logger.info(f"Incoming call from {caller_number} to {called_number} (SID: {call_sid})")
        log_call_event(call_sid, caller_number, called_number, "call_started")

        host = request.headers.get("host") or request.url.netloc
        stream_url = f"wss://{host}/media-stream"

        logger.info(f"Media Stream URL: {stream_url}")

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
        "from_number": "Unknown",
        "to_number": "Unknown",
        "call_sid": "Unknown",
        "transcripts": [],
    }

    try:
        logger.info("Connecting to OpenAI Realtime API (MINI MODEL)...")
        async with websockets.connect(
            # Using gpt-4o-mini-realtime-preview for 60% cost savings
            f"wss://api.openai.com/v1/realtime?"
            f"model=gpt-4o-mini-realtime-preview&temperature={TEMPERATURE}",
            additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
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

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"Error in /media-stream handler: {e}")
    finally:
        # Log call summary before cleanup
        logger.info(f"🧹 FINALLY BLOCK ENTERED for session: {session_id}")
        try:
            logger.info(f"Starting cleanup for session: {session_id}")
            session_data = ACTIVE_SESSIONS.get(session_id, {})
            transcripts = session_data.get("transcripts", [])
            call_sid = session_data.get("call_sid", "Unknown")
            from_number = session_data.get("from_number", "Unknown")
            to_number = session_data.get("to_number", "Unknown")

            logger.info(f"Found {len(transcripts)} transcripts for session {session_id}")

            # Calculate call duration
            start_time = session_data.get("start_time", time.time())
            duration = int(time.time() - start_time)

            if transcripts:
                summary = generate_call_summary(transcripts)
                logger.info(f"Call summary: {summary}")
                log_call_event(
                    call_sid=call_sid,
                    from_number=from_number,
                    to_number=to_number,
                    event_type="call_ended",
                    data={
                        "summary": summary,
                        "transcript_count": len(transcripts),
                        "duration": duration
                    }
                )
            else:
                logger.info("No transcripts found, logging with default summary")
                log_call_event(
                    call_sid=call_sid,
                    from_number=from_number,
                    to_number=to_number,
                    event_type="call_ended",
                    data={
                        "summary": "No conversation recorded",
                        "duration": duration,
                        "transcript_count": 0
                    }
                )

            ACTIVE_SESSIONS.pop(session_id, None)
            logger.info(f"Session ended: {session_id}")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


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
    Configure the Realtime session using PCM16 format.
    We convert between Twilio's μ-law and OpenAI's PCM16.
    """
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-4o-mini-realtime-preview",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE,
                },
            },
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
                call_sid = data["start"].get("callSid", "Unknown")
                custom_params = data["start"].get("customParameters", {})

                ACTIVE_SESSIONS[session_id]["stream_sid"] = stream_sid
                ACTIVE_SESSIONS[session_id]["call_sid"] = call_sid
                ACTIVE_SESSIONS[session_id]["from_number"] = custom_params.get("from_number", "Unknown")
                ACTIVE_SESSIONS[session_id]["to_number"] = custom_params.get("to_number", "Unknown")

                logger.info(f"Twilio stream started - SID: {stream_sid}, CallSID: {call_sid}")

            elif event_type == "media":
                # Twilio sends base64-encoded G.711 μ-law audio in data['media']['payload']
                try:
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": data["media"]["payload"],
                    }
                    await openai_ws.send(json.dumps(audio_append))
                except Exception as e:
                    logger.debug(f"Error sending audio to OpenAI: {e}")
                    break

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
            if (event_type == "response.audio.delta" or event_type == "response.output_audio.delta") and response.get("delta"):
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
                    # Store in session for summary
                    if session_id in ACTIVE_SESSIONS:
                        ACTIVE_SESSIONS[session_id]["transcripts"].append({
                            "role": "assistant",
                            "content": transcript
                        })

            # User speech transcript
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript") or ""
                if transcript:
                    logger.info(f"User said: {transcript}")
                    # Store in session for summary
                    if session_id in ACTIVE_SESSIONS:
                        ACTIVE_SESSIONS[session_id]["transcripts"].append({
                            "role": "user",
                            "content": transcript
                        })

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

    logger.info(f"Starting Gabriel Pascual's Voice Agent on port {PORT}")
    logger.info("Using model: gpt-4o-mini-realtime-preview (60% cost savings)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
