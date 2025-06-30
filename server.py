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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/pascual_bot/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ['OPENAI_API_KEY', 'TWILIO_SID', 'TWILIO_TOKEN', 'N8N_WEBHOOK_URL']
missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    logger.critical(f"Missing required env vars: {missing_vars}")
    sys.exit(1)

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
TWILIO_SID      = os.getenv('TWILIO_SID')
TWILIO_TOKEN    = os.getenv('TWILIO_TOKEN')
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL')
PORT            = int(os.getenv('PORT', 5024))
VOICE           = os.getenv('VOICE', 'sage')
MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', 300))  # seconds

# Updated system message to be more concise
SYSTEM_MESSAGE = (
    "You are a professional sales representative for Pascual & Co. Be concise and direct:\n"
    "1. Greet callers briefly: 'Hi, this is [Your Name] from Pascual & Co. Who am I speaking with?'\n"
    "2. Listen to their needs and explain services in 1-2 sentences max.\n" 
    "3. Ask direct questions. Keep responses under 20 words when possible.\n"
    "4. Avoid lengthy explanations unless specifically asked.\n"
    "5. Focus on next steps and actionable items.\n"
    "Keep all responses brief, professional, and to the point."
)

# ---------------- helpers ----------------
def is_voiced(ulaw: bytes) -> bool:
    if not VAD_ENABLED:
        return True
    try:
        pcm16 = audioop.ulaw2lin(ulaw, 2)
        return VAD.is_speech(pcm16, SAMPLE_RATE)
    except Exception as e:
        logger.debug(f"VAD error: {e}")
        return True

# Session tracking with better cleanup
ACTIVE_SESSIONS = {}
CALLER_NUMBERS = {}
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down, cleaning up sessions...")
    # Create a list copy to avoid modification during iteration
    session_ids = list(ACTIVE_SESSIONS.keys())
    if session_ids:
        logger.info(f"Cleaning up {len(session_ids)} active sessions")
        cleanup_tasks = [cleanup_session(sid) for sid in session_ids]
        try:
            await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Session cleanup timed out")
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Pascual & Co Assistant",
    description="AI-powered assistant for Pascual & Co",
    version="1.0.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# -------------- summary util --------------
def extract_call_summary(conversation_text: str, caller_name="", caller_number="") -> dict:
    if not conversation_text.strip():
        return {
            "caller_name": caller_name,      
            "caller_number": caller_number,
            "call_summary": "No conversation content available",
            "call_duration": "Unknown",      
            "key_topics": [],
            "follow_up_needed": False,
        }
    
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""Analyze this sales call and extract key information.

Conversation:
{conversation_text}

Return JSON with: caller_name, caller_number, call_summary, call_duration, key_topics (array), follow_up_needed (boolean)"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, 
            max_tokens=300, 
            timeout=15,
        )
        
        raw_content = response.choices[0].message.content.strip()
        # Clean up markdown formatting
        raw_content = re.sub(r"```(?:json)?|```", "", raw_content, flags=re.I).strip()
        
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        logger.error(f"Error extracting call summary: {e}")
        data = {}
    
    # Default values with update
    default = {
        "caller_name": caller_name or "Unknown",
        "caller_number": caller_number or "Unknown",
        "call_summary": "Call completed",
        "call_duration": "Unknown",
        "key_topics": [], 
        "follow_up_needed": False,
    }
    
    # Only update with valid keys
    for key, value in data.items():
        if key in default:
            default[key] = value
            
    return default

async def send_to_n8n(call_data: dict):
    try:
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(N8N_WEBHOOK_URL, json=call_data)
            logger.info(f"Sent call data to n8n: {response.status_code} - Data: {json.dumps(call_data, indent=2)}")
            return True
    except Exception as e:
        logger.error(f"Failed to send data to n8n: {e}")
        return False

async def cleanup_session(sid: str):
    session = ACTIVE_SESSIONS.pop(sid, None)
    if not session: 
        return
    
    logger.info(f"Cleaning up session {sid}")
    
    try:
        conversation_text = "\n".join(session["conversation"])
        
        # Ensure we have the caller number
        caller_number = session.get("caller_number", "Unknown")
        logger.info(f"Session {sid} cleanup - Caller number: {caller_number}")
        
        summary = extract_call_summary(
            conversation_text,
            session.get("caller_name", ""), 
            caller_number,
        )
        
        # Force the caller number to be included
        summary["caller_number"] = caller_number
        
        summary.update({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "session_duration": time.time() - session.get("start_time", time.time()),
            "total_messages": len(session["conversation"]),
            "call_sid": session.get("call_sid", "unknown"),
            "company": "Pascual & Co", 
            "call_type": "inbound_sales",
        })
        
        logger.info(f"Final summary for session {sid}: {json.dumps(summary, indent=2)}")
        
        # Send to n8n with timeout
        await asyncio.wait_for(send_to_n8n(summary), timeout=10)
        
    except asyncio.TimeoutError:
        logger.warning(f"n8n webhook timeout for session {sid}")
    except Exception as e:
        logger.error(f"Error in cleanup_session for {sid}: {e}")

# ---------------- routes ----------------
@app.get("/", response_class=JSONResponse)
async def index(): 
    return {
        "message": "Pascual & Co Bot", 
        "version": app.version,
        "status": "running",
        "active_sessions": len(ACTIVE_SESSIONS)
    }

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    try:
        params = await request.form() if request.method == "POST" else request.query_params
        caller = params.get("From", "Unknown")
        call_sid = params.get("CallSid")
        
        logger.info(f"Incoming call from {caller}, CallSid: {call_sid}")
        
        # Store caller number immediately
        if call_sid and caller != "Unknown":
            CALLER_NUMBERS[call_sid] = caller
            logger.info(f"Stored caller number {caller} for CallSid {call_sid}")
        
        # Use the actual hostname from the request
        host = request.headers.get("host", request.url.hostname)
        
        resp = VoiceResponse()
        connect = Connect()
        connect.stream(url=f"wss://{host}/media-stream")
        resp.append(connect)
        
        return HTMLResponse(str(resp), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error in incoming_call: {e}")
        # Return a basic TwiML response as fallback
        resp = VoiceResponse()
        resp.say("Sorry, we're experiencing technical difficulties. Please try again later.")
        return HTMLResponse(str(resp), media_type="application/xml")

# -------------- WebSocket bridge ---------------
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"WebSocket connection attempt: {session_id}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established: {session_id}")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}")
        return
    
    # Shared state between tasks
    shared_state = {
        "stream_sid": None,
        "call_sid": None,
        "greeting_sent": False,
        "openai_ws": None
    }

    # Initialize session
    ACTIVE_SESSIONS[session_id] = {
        "conversation": [], 
        "caller_name": "", 
        "caller_number": "Unknown",
        "call_sid": None, 
        "start_time": start_time, 
        "last_activity": time.time(),
    }

    try:
        # Connect to OpenAI with timeout
        logger.info(f"Connecting to OpenAI for session {session_id}")
        
        openai_ws = await asyncio.wait_for(
            websockets.connect(
                "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}", 
                    "OpenAI-Beta": "realtime=v1"
                },
                ping_interval=20, 
                ping_timeout=10,
                close_timeout=10
            ),
            timeout=30
        )
        
        shared_state["openai_ws"] = openai_ws
        logger.info(f"OpenAI connection established for session {session_id}")
        
        # Configure OpenAI session with emphasis on being concise
        await openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "voice": VOICE,
                "instructions": SYSTEM_MESSAGE,
                "modalities": ["text", "audio"],
                "temperature": 0.4,  # Slightly lower for more consistent responses
                "input_audio_transcription": {"model": "whisper-1"},
            }
        }))

        async def send_greeting():
            if shared_state["greeting_sent"] or not openai_ws.open:
                return
            try:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", 
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Please give a brief greeting as instructed - keep it under 15 words."}],
                    },
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))
                shared_state["greeting_sent"] = True
                logger.info(f"Greeting sent for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to send greeting for session {session_id}: {e}")

        async def handle_twilio_messages():
            try:
                while True:
                    # Check for shutdown
                    if shutdown_event.is_set():
                        logger.info(f"Shutdown requested, closing Twilio handler for {session_id}")
                        break
                    
                    try:
                        # Use timeout to avoid hanging
                        message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        data = json.loads(message)
                        event = data.get("event")
                        
                        if event == "start":
                            shared_state["stream_sid"] = data["start"]["streamSid"]
                            shared_state["call_sid"] = data["start"].get("callSid")
                            
                            # Get caller number from stored data
                            caller = CALLER_NUMBERS.get(shared_state["call_sid"], "Unknown")
                            
                            # Update session with caller info
                            ACTIVE_SESSIONS[session_id].update({
                                "call_sid": shared_state["call_sid"],
                                "caller_number": caller
                            })
                            
                            logger.info(f"Stream started - StreamSid: {shared_state['stream_sid']}, CallSid: {shared_state['call_sid']}, Caller: {caller}")
                            
                            # Small delay then send greeting
                            await asyncio.sleep(1.0)
                            await send_greeting()
                            
                        elif event == "media" and openai_ws.open:
                            payload = data["media"]["payload"]
                            audio_data = base64.b64decode(payload)
                            
                            if is_voiced(audio_data):
                                await openai_ws.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": payload
                                }))
                                
                        elif event == "stop":
                            logger.info(f"Twilio stream stopped for session {session_id}")
                            break
                            
                    except asyncio.TimeoutError:
                        # Continue loop, this is normal
                        continue
                    except WebSocketDisconnect:
                        logger.info(f"Twilio WebSocket disconnected for session {session_id}")
                        break
                        
            except Exception as e:
                logger.error(f"Error in Twilio message handler for session {session_id}: {e}")

        async def handle_openai_messages():
            try:
                while openai_ws.open:
                    # Check for shutdown
                    if shutdown_event.is_set():
                        logger.info(f"Shutdown requested, closing OpenAI handler for {session_id}")
                        break
                    
                    try:
                        message = await asyncio.wait_for(openai_ws.recv(), timeout=1.0)
                        response = json.loads(message)
                        
                        ACTIVE_SESSIONS[session_id]["last_activity"] = time.time()

                        response_type = response.get("type")
                        
                        if response_type == "conversation.item.input_audio_transcription.completed":
                            transcript = response.get("transcript", "").strip()
                            confidence = response.get("confidence", 1.0)
                            
                            if not transcript or confidence < 0.6:
                                continue
                                
                            words = transcript.split()
                            if len(words) < 2:  # Skip very short utterances
                                continue
                                
                            ACTIVE_SESSIONS[session_id]["conversation"].append(f"Caller: {transcript}")
                            logger.info(f"Session {session_id} - Caller: {transcript}")

                            # Extract caller name
                            if not ACTIVE_SESSIONS[session_id]["caller_name"]:
                                name_match = re.search(r"(?:i'm|i am|my name is|this is|it's)\s+([a-zA-Z]+)", transcript.lower())
                                if name_match:
                                    name = name_match.group(1).title()
                                    ACTIVE_SESSIONS[session_id]["caller_name"] = name
                                    logger.info(f"Extracted caller name: {name}")

                        elif response_type == "response.audio_transcript.done":
                            transcript = response.get("transcript", "")
                            if transcript:
                                ACTIVE_SESSIONS[session_id]["conversation"].append(f"Assistant: {transcript}")
                                logger.info(f"Session {session_id} - Assistant: {transcript}")

                        elif response_type == "response.audio.delta":
                            delta = response.get("delta")
                            if delta and shared_state["stream_sid"]:
                                try:
                                    await websocket.send_json({
                                        "event": "media",
                                        "streamSid": shared_state["stream_sid"],
                                        "media": {
                                            "payload": base64.b64encode(
                                                base64.b64decode(delta)
                                            ).decode()
                                        },
                                    })
                                except Exception as e:
                                    logger.error(f"Failed to send audio to Twilio for session {session_id}: {e}")
                                    break
                                    
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"OpenAI WebSocket closed for session {session_id}")
                        break
                        
            except Exception as e:
                logger.error(f"Error in OpenAI message handler for session {session_id}: {e}")

        # Run both handlers concurrently with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    handle_twilio_messages(),
                    handle_openai_messages(),
                    return_exceptions=True
                ),
                timeout=MAX_CALL_DURATION
            )
        except asyncio.TimeoutError:
            logger.warning(f"Session {session_id} timed out after {MAX_CALL_DURATION} seconds")

    except asyncio.TimeoutError:
        logger.error(f"OpenAI connection timeout for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup
        logger.info(f"Cleaning up session {session_id}")
        
        # Close OpenAI connection
        if shared_state["openai_ws"] and not shared_state["openai_ws"].closed:
            try:
                await asyncio.wait_for(shared_state["openai_ws"].close(), timeout=5)
            except:
                pass
        
        # Close Twilio connection
        try:
            await websocket.close()
        except:
            pass
            
        # Clean up session data
        await cleanup_session(session_id)
        
        # Clean up caller number
        if shared_state["call_sid"]:
            CALLER_NUMBERS.pop(shared_state["call_sid"], None)
            
        logger.info(f"Session {session_id} cleanup complete")

# ------------- health / monitoring -------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "vad_enabled": VAD_ENABLED,
        "version": app.version,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
    }

@app.get("/metrics")
async def metrics():
    return {
        "active_sessions": len(ACTIVE_SESSIONS),
        "total_callers": len(CALLER_NUMBERS),
        "memory_usage": f"{os.getpid()}",  # Process ID for monitoring
        "timestamp": dt.datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, 
        content={"message": "Internal server error", "path": str(request.url)}
    )

# Track start time
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Pascual & Co Bot on port {PORT}")
    logger.info(f"VAD Enabled: {VAD_ENABLED}")
    logger.info(f"Voice: {VOICE}")
    logger.info(f"Max call duration: {MAX_CALL_DURATION} seconds")
    
    # Configure uvicorn with better settings for production
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server shutdown complete")