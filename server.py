import os, json, base64, asyncio, websockets, re, time, uuid, datetime as dt, logging, httpx, audioop
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import signal
import sys
import webrtcvad

load_dotenv()

# Minimal logging, only critical system errors
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Configuration with validation
REQUIRED_ENV_VARS = ['OPENAI_API_KEY', 'TWILIO_SID', 'TWILIO_TOKEN', 'N8N_WEBHOOK_URL']
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"Missing required environment variables: {missing_vars}")
    sys.exit(1)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL')
PORT = int(os.getenv('PORT', 5024))
VOICE = os.getenv('VOICE', 'sage')
MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', 300))  # 5 minutes default

# ------------------
#  VAD configuration
# ------------------
VAD_MODE = 2  # 0‑3, higher is more aggressive at filtering noise
VAD = webrtcvad.Vad(VAD_MODE)
SAMPLE_RATE = 8000  # Twilio streams μ‑law 8 kHz
FRAME_MS = 20       # Twilio sends 20 ms frames


def is_voiced(ulaw_bytes: bytes) -> bool:
    """Return True if the audio frame contains speech."""
    try:
        pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)  # convert to 16‑bit linear PCM
        return VAD.is_speech(pcm16, SAMPLE_RATE)
    except Exception:
        return False

# Sales‑focused system message for Zapstrix
SYSTEM_MESSAGE = (
    "You are a professional sales representative for Zapstrix. Your role is to:\n"
    "1. Warmly greet callers and introduce yourself as representing Zapstrix.\n"
    "2. Listen to their needs and inquiries about our services.\n"
    "3. Provide helpful information about what Zapstrix offers.\n"
    "   Zapstrix is an AI Agent and Automation company that specializes in building custom solutions for businesses of all sizes.\n"
    "   We focus on delivering high‑quality, scalable, and secure solutions tailored to meet the unique needs of our clients,\n"
    "   improving lead generation, customer support, and operational efficiency.\n"
    "4. Build rapport and trust through professional, friendly conversation.\n"
    "5. Ask for the caller's name naturally during the conversation.\n"
    "6. Answer questions about our company and services professionally.\n"
    "7. Handle objections with empathy and expertise.\n"
    "8. Keep conversations focused but natural — you're here to help and inform.\n"
    "Always be conversational, professional, and genuinely helpful. Focus on understanding their needs rather than being pushy.\n"
    "Start by saying: \"Hello! Thank you for calling Zapstrix. I'm here to help you today. May I ask who I'm speaking with?\""
)

# Session storage for conversation tracking
ACTIVE_SESSIONS = {}
CALLER_NUMBERS = {}

# Graceful shutdown handling
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
after = lifespan
async def lifespan(app: FastAPI):
    yield
    # Cleanup active sessions on shutdown
    for session_id in list(ACTIVE_SESSIONS.keys()):
        try:
            await cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

app = FastAPI(
    title="Zapstrix Assistant",
    description="AI‑powered assistant for Zapstrix",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
#  Helper / utility funcs
# ------------------------

def extract_call_summary(conversation_text: str, caller_name: str = "", caller_number: str = "") -> dict:
    """Extract call information using OpenAI API with proper error handling"""
    if not conversation_text.strip():
        return {
            "caller_name": caller_name,
            "caller_number": caller_number,
            "call_summary": "No conversation content available",
            "call_duration": "Unknown",
            "key_topics": [],
            "follow_up_needed": False
        }

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""Analyze this sales call conversation and extract key information.\n\nConversation:\n{conversation_text}\n\nReturn ONLY valid JSON with these exact keys:\n- \"caller_name\": The caller's name (use \"{caller_name}\" if provided, otherwise extract from conversation)\n- \"caller_number\": \"{caller_number}\"\n- \"call_summary\": Brief summary of the call (2‑3 sentences)\n- \"call_duration\": Estimate in minutes\n- \"key_topics\": Array of main topics discussed\n- \"follow_up_needed\": Boolean indicating if follow‑up is recommended\n\nJSON:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            timeout=10,
        )

        raw_response = response.choices[0].message.content.strip()
        raw_response = re.sub(r"```(?:json)?|```", "", raw_response, flags=re.I).strip()
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)

        default_data = {
            "caller_name": caller_name or "Unknown",
            "caller_number": caller_number or "Unknown",
            "call_summary": "Call completed successfully",
            "call_duration": "Unknown",
            "key_topics": [],
            "follow_up_needed": False,
        }

        if json_match:
            data = json.loads(json_match.group(0))
            default_data.update({k: v for k, v in data.items() if k in default_data})
        return default_data

    except Exception:
        return {
            "caller_name": caller_name or "Unknown",
            "caller_number": caller_number or "Unknown",
            "call_summary": "Call completed — summary extraction failed",
            "call_duration": "Unknown",
            "key_topics": ["Summary extraction failed"],
            "follow_up_needed": True,
        }


async def send_to_n8n(call_data: dict, call_sid: str):
    """Send call data to n8n webhook with retry logic"""
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            payload = {
                **call_data,
                "timestamp": dt.datetime.now().isoformat(),
                "call_sid": call_sid,
                "company": "Zapstrix",
                "call_type": "inbound_sales",
            }
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(N8N_WEBHOOK_URL, json=payload)
                if response.status_code < 400:
                    return True
        except Exception:
            pass
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
    return False


async def cleanup_session(session_id: str):
    """Clean up session data and send final summary to N8N"""
    if session_id not in ACTIVE_SESSIONS:
        return

    session = ACTIVE_SESSIONS[session_id]
    try:
        conversation_text = "\n".join(session["conversation"])
        if conversation_text.strip():
            call_data = extract_call_summary(
                conversation_text,
                session.get("caller_name", ""),
                session.get("caller_number", "Unknown"),
            )
            call_data["session_duration"] = time.time() - session.get("start_time", time.time())
            call_data["total_messages"] = len(session["conversation"])
            await send_to_n8n(call_data, session.get("call_sid", "unknown"))
    finally:
        ACTIVE_SESSIONS.pop(session_id, None)


# ---------------
#     ROUTES
# ---------------

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Zapstrix Bot — AI Assistant", "version": "1.1.0", "status": "operational"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Return TwiML that starts the WebSocket stream."""
    try:
        form_data = await request.form() if request.method == "POST" else {}
        caller_number = form_data.get("From") or request.query_params.get("From", "Unknown")
        call_sid = form_data.get("CallSid") or request.query_params.get("CallSid")

        if call_sid and caller_number != "Unknown":
            CALLER_NUMBERS[call_sid] = caller_number

        response = VoiceResponse()
        host = request.url.hostname
        connect = Connect()
        connect.stream(url=f"wss://{host}/media-stream")
        response.append(connect)
        return HTMLResponse(content=str(response), media_type="application/xml")

    except Exception:
        response = VoiceResponse()
        response.say("We're sorry — technical difficulties. Please try again later.")
        return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    await websocket.accept()
    call_start_time = time.time()

    try:
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=30,
            ping_timeout=10,
        ) as openai_ws:

            ACTIVE_SESSIONS[session_id] = {
                "conversation": [],
                "caller_name": "",
                "caller_number": "Unknown",
                "call_sid": None,
                "start_time": call_start_time,
                "last_activity": time.time(),
            }
            await send_session_update(openai_ws)

            stream_sid = None
            call_sid = None

            async def send_initial_greeting():
                await asyncio.sleep(1)
                if openai_ws.open:
                    await openai_ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": "Please greet the caller as instructed."}
                                    ],
                                },
                            }
                        )
                    )
                    await openai_ws.send(json.dumps({"type": "response.create"}))

            async def receive_from_twilio():
                nonlocal stream_sid, call_sid
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        event = data.get("event")

                        if event == "media" and openai_ws.open:
                            payload_b64 = data["media"]["payload"]
                            ulaw_bytes = base64.b64decode(payload_b64)
                            if is_voiced(ulaw_bytes):
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": payload_b64,
                                }
                                await openai_ws.send(json.dumps(audio_append))

                        elif event == "start":
                            stream_sid = data["start"]["streamSid"]
                            call_sid = data["start"].get("callSid")
                            caller_number = CALLER_NUMBERS.get(call_sid, "Unknown")
                            ACTIVE_SESSIONS[session_id].update(
                                {"call_sid": call_sid, "caller_number": caller_number}
                            )
                            asyncio.create_task(send_initial_greeting())

                        elif event == "stop":
                            asyncio.create_task(cleanup_session(session_id))
                            break
                except WebSocketDisconnect:
                    asyncio.create_task(cleanup_session(session_id))
                finally:
                    CALLER_NUMBERS.pop(call_sid, None)

            async def send_to_twilio():
                nonlocal stream_sid
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)

                        # Track last activity
                        ACTIVE_SESSIONS[session_id]["last_activity"] = time.time()

                        # User transcription
                        if (
                            response["type"]
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            transcript = response.get("transcript", "")
                            confidence = response.get("confidence", 1.0)
                            # Ignore low‑confidence or super‑short noise transcripts
                            if confidence < 0.6 or len(transcript.split()) < 3:
                                continue
                            ACTIVE_SESSIONS[session_id]["conversation"].append(
                                f"Caller: {transcript}"
                            )
                            # Name extraction
                            if not ACTIVE_SESSIONS[session_id]["caller_name"]:
                                match = re.search(
                                    r"(?:i'm|i am|my name is|this is|it's)\s+([a-zA-Z]+)",
                                    transcript.lower(),
                                )
                                if match:
                                    ACTIVE_SESSIONS[session_id]["caller_name"] = match.group(1).title()

                        # Assistant transcription
                        if response["type"] == "response.audio_transcript.done":
                            transcript = response.get("transcript", "")
                            if transcript:
                                ACTIVE_SESSIONS[session_id]["conversation"].append(
                                    f"Assistant: {transcript}"
                                )

                        # Audio back to Twilio
                        if (
                            response["type"] == "response.audio.delta" and response.get("delta") and stream_sid
                        ):
                            try:
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": base64.b64encode(
                                            base64.b64decode(response["delta"])
                                        ).decode("utf-8")
                                    },
                                }
                                await websocket.send_json(audio_delta)
                            except Exception:
                                pass
                except Exception:
                    pass

            await asyncio.gather(receive_from_twilio(), send_to_twilio())
    finally:
        if session_id in ACTIVE_SESSIONS:
            await cleanup_session(session_id)


async def send_session_update(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad", "sensitivity": "medium"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.7,
            "input_audio_transcription": {"model": "whisper-1"},
        },
    }
    await openai_ws.send(json.dumps(session_update))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "n8n_configured": bool(N8N_WEBHOOK_URL),
        "uptime": time.time(),
        "version": "1.1.0",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"message": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="error",
            access_log=False,
        )
    except KeyboardInterrupt:
        pass
