import os, json, base64, asyncio, websockets, re, time, uuid, datetime as dt, logging, httpx, audioop
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import signal, sys

# ---------- optional VAD ----------
try:
    import webrtcvad

    VAD = webrtcvad.Vad(2)          # 0–3, higher = stricter
    SAMPLE_RATE = 8000              # Twilio μ-law 8 kHz
    VAD_ENABLED = True
except ImportError:
    VAD_ENABLED = False
# ----------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)  # Changed from CRITICAL to INFO for better debugging
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ['OPENAI_API_KEY', 'TWILIO_SID', 'TWILIO_TOKEN', 'N8N_WEBHOOK_URL']
missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    print("Missing required env vars:", missing_vars); sys.exit(1)

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
TWILIO_SID      = os.getenv('TWILIO_SID')
TWILIO_TOKEN    = os.getenv('TWILIO_TOKEN')
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL')
PORT            = int(os.getenv('PORT', 5024))
VOICE           = os.getenv('VOICE', 'sage')
MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', 300))  # seconds

SYSTEM_MESSAGE = (
    "You are a professional sales representative for Zapstrix. Your role is to:\n"
    "1. Warmly greet callers and introduce yourself.\n"
    "2. Listen to their needs and explain Zapstrix's AI-driven automations.\n"
    "3. Build rapport, handle objections, stay helpful.\n"
    'Start with: "Hello! Thank you for calling Zapstrix. I\'m here to help you today. May I ask who I\'m speaking with?"'
)

# ---------------- helpers ----------------
def is_voiced(ulaw: bytes) -> bool:
    if not VAD_ENABLED:
        return True
    try:
        pcm16 = audioop.ulaw2lin(ulaw, 2)
        return VAD.is_speech(pcm16, SAMPLE_RATE)
    except Exception:
        return True

# Session tracking
ACTIVE_SESSIONS, CALLER_NUMBERS = {}, {}
shutdown_event = asyncio.Event()
signal.signal(signal.SIGINT,  lambda *_: shutdown_event.set())
signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for sid in list(ACTIVE_SESSIONS):
        await cleanup_session(sid)

app = FastAPI(
    title="Zapstrix Assistant",
    description="AI-powered assistant for Zapstrix",
    version="1.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------- summary util --------------
def extract_call_summary(conversation_text: str, caller_name="", caller_number="") -> dict:
    if not conversation_text.strip():
        return {
            "caller_name": caller_name,      "caller_number": caller_number,
            "call_summary": "No conversation content available",
            "call_duration": "Unknown",      "key_topics": [],
            "follow_up_needed": False,
        }
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""Analyze this sales call and extract:
caller_name, caller_number, call_summary, call_duration, key_topics, follow_up_needed.

Conversation:
{conversation_text}

Return as JSON with the following structure:
{{
    "caller_name": "string",
    "caller_number": "string", 
    "call_summary": "string",
    "call_duration": "string",
    "key_topics": ["topic1", "topic2"],
    "follow_up_needed": boolean
}}"""
        
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300, timeout=10,
        )
        raw = rsp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw, flags=re.I).strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", raw, re.S)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = {}
    except Exception as e:
        logger.error(f"Error extracting call summary: {e}")
        data = {}
    
    default = {
        "caller_name": caller_name or "Unknown",
        "caller_number": caller_number or "Unknown",
        "call_summary": "Call completed",
        "call_duration": "Unknown",
        "key_topics": [], 
        "follow_up_needed": False,
    }
    default.update({k: v for k, v in data.items() if k in default})
    return default

async def send_to_n8n(call_data: dict):
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            response = await c.post(N8N_WEBHOOK_URL, json=call_data)
            logger.info(f"Sent call data to n8n: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send data to n8n: {e}")

async def cleanup_session(sid: str):
    ses = ACTIVE_SESSIONS.pop(sid, None)
    if not ses: 
        return
    
    logger.info(f"Cleaning up session {sid}")
    
    summary = extract_call_summary(
        "\n".join(ses["conversation"]),
        ses.get("caller_name", ""), 
        ses.get("caller_number", "Unknown"),
    )
    
    summary.update({
        "timestamp": dt.datetime.utcnow().isoformat(),
        "session_duration": time.time() - ses.get("start_time", time.time()),
        "total_messages": len(ses["conversation"]),
        "call_sid": ses.get("call_sid", "unknown"),
        "company": "Zapstrix", 
        "call_type": "inbound_sales",
    })
    
    await send_to_n8n(summary)

# ---------------- routes ----------------
@app.get("/", response_class=JSONResponse)
async def index(): 
    return {"message": "Zapstrix Bot", "version": app.version}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming(request: Request):
    params = await request.form() if request.method == "POST" else request.query_params
    caller = params.get("From", "Unknown")
    callSid = params.get("CallSid")
    
    logger.info(f"Incoming call from {caller}, CallSid: {callSid}")
    
    if callSid and caller != "Unknown":
        CALLER_NUMBERS[callSid] = caller
    
    host = request.url.hostname
    resp = VoiceResponse()
    conn = Connect()
    conn.stream(url=f"wss://{host}/media-stream")
    resp.append(conn)
    
    return HTMLResponse(str(resp), media_type="application/xml")

# -------------- WebSocket bridge ---------------
@app.websocket("/media-stream")
async def media(ws: WebSocket):
    sid = str(uuid.uuid4())
    await ws.accept()
    start = time.time()
    
    logger.info(f"WebSocket connection established: {sid}")
    
    # Shared variables between tasks
    shared_state = {
        "streamSid": None,
        "callSid": None,
        "greeting_sent": False
    }

    try:
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}", 
                "OpenAI-Beta": "realtime=v1"
            },
            ping_interval=30, 
            ping_timeout=10
        ) as ai:

            # --- initialise session ---
            ACTIVE_SESSIONS[sid] = {
                "conversation": [], 
                "caller_name": "", 
                "caller_number": "Unknown",
                "call_sid": None, 
                "start_time": start, 
                "last_activity": time.time(),
            }
            
            # Configure OpenAI session
            await ai.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "voice": VOICE,
                    "instructions": SYSTEM_MESSAGE,
                    "modalities": ["text", "audio"],
                    "temperature": 0.7,
                    "input_audio_transcription": {"model": "whisper-1"},
                }
            }))

            # --- greeting function ---
            async def send_greeting():
                if shared_state["greeting_sent"] or not ai.open:
                    return
                try:
                    await ai.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message", 
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Please greet the caller as instructed."}],
                        },
                    }))
                    await ai.send(json.dumps({"type": "response.create"}))
                    shared_state["greeting_sent"] = True
                    logger.info("Sent greeting to OpenAI")
                except Exception as e:
                    logger.error(f"Failed to send greeting: {e}")

            # ---------- inner tasks ----------
            async def twilio_reader():
                try:
                    async for msg in ws.iter_text():
                        data = json.loads(msg)
                        evt = data.get("event")
                        
                        if evt == "start":
                            shared_state["streamSid"] = data["start"]["streamSid"]
                            shared_state["callSid"] = data["start"].get("callSid")
                            caller = CALLER_NUMBERS.get(shared_state["callSid"], "Unknown")
                            
                            ACTIVE_SESSIONS[sid].update({
                                "call_sid": shared_state["callSid"],
                                "caller_number": caller
                            })
                            
                            logger.info(f"Stream started - StreamSid: {shared_state['streamSid']}, CallSid: {shared_state['callSid']}")
                            
                            # Add small delay to ensure connection is stable
                            await asyncio.sleep(0.5)
                            await send_greeting()
                            
                        elif evt == "media" and ai.open:
                            b64 = data["media"]["payload"]
                            ulaw = base64.b64decode(b64)
                            if is_voiced(ulaw):
                                await ai.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": b64
                                }))
                                
                        elif evt == "stop":
                            logger.info("Twilio stream stopped")
                            break
                            
                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                except Exception as e:
                    logger.error(f"Error in twilio_reader: {e}")
                finally:
                    await cleanup_session(sid)
                    if shared_state["callSid"]:
                        CALLER_NUMBERS.pop(shared_state["callSid"], None)

            async def openai_reader():
                try:
                    async for raw in ai:
                        res = json.loads(raw)
                        ACTIVE_SESSIONS[sid]["last_activity"] = time.time()

                        # --- handle caller transcription ---
                        if res["type"] == "conversation.item.input_audio_transcription.completed":
                            transcript = res.get("transcript", "")
                            confidence = res.get("confidence", 1.0)
                            words = transcript.strip().split()
                            word_cnt = len(words)

                            # ---- smarter short-utterance filter ----
                            if confidence < 0.6:
                                continue
                                
                            expecting_short = False
                            if ACTIVE_SESSIONS[sid]["conversation"]:
                                last_bot = ACTIVE_SESSIONS[sid]["conversation"][-1]
                                expecting_short = any(
                                    phrase in last_bot.lower()
                                    for phrase in ["your name", "yes or no", "press", "say yes"]
                                )
                                
                            looks_like_name = (
                                word_cnt == 1 and words[0].istitle()
                                and 2 <= len(words[0]) <= 12 and words[0].isalpha()
                            )
                            
                            if word_cnt < 3 and not (expecting_short or looks_like_name):
                                continue

                            ACTIVE_SESSIONS[sid]["conversation"].append(f"Caller: {transcript}")
                            logger.info(f"Caller said: {transcript}")

                            # Extract name if not captured
                            if not ACTIVE_SESSIONS[sid]["caller_name"]:
                                m = re.search(r"(?:i'm|i am|my name is|this is|it's)\s+([a-zA-Z]+)", transcript.lower())
                                if m:
                                    name = m.group(1).title()
                                    ACTIVE_SESSIONS[sid]["caller_name"] = name
                                    logger.info(f"Extracted caller name: {name}")

                        # --- assistant transcript ---
                        elif res["type"] == "response.audio_transcript.done":
                            transcript = res.get("transcript", "")
                            if transcript:
                                ACTIVE_SESSIONS[sid]["conversation"].append(f"Assistant: {transcript}")
                                logger.info(f"Assistant said: {transcript}")

                        # --- send audio back to Twilio ---
                        elif res["type"] == "response.audio.delta" and res.get("delta"):
                            if shared_state["streamSid"]:  # Make sure we have streamSid
                                try:
                                    await ws.send_json({
                                        "event": "media",
                                        "streamSid": shared_state["streamSid"],
                                        "media": {
                                            "payload": base64.b64encode(
                                                base64.b64decode(res["delta"])
                                            ).decode()
                                        },
                                    })
                                except Exception as e:
                                    logger.error(f"Failed to send audio to Twilio: {e}")
                                    
                except Exception as e:
                    logger.error(f"Error in openai_reader: {e}")

            await asyncio.gather(twilio_reader(), openai_reader())
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Ensure cleanup happens
        if sid in ACTIVE_SESSIONS:
            await cleanup_session(sid)
        logger.info(f"Session {sid} ended")

# ------------- health / error -------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "vad_enabled": VAD_ENABLED,
        "version": app.version,
        "timestamp": dt.datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500, 
        content={"message": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Zapstrix Bot on port {PORT}")
    print(f"VAD Enabled: {VAD_ENABLED}")
    print(f"Voice: {VOICE}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT, 
        log_level="info",  # Changed from error to info
        access_log=True    # Changed from False to True
    )