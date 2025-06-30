import os, json, base64, asyncio, websockets, re, time, uuid, datetime as dt, logging, httpx
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import signal
import sys

load_dotenv()

# Enhanced logging for debugging
logging.basicConfig(level=logging.INFO)
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

# Updated system message for better responses
SYSTEM_MESSAGE = """You are a professional sales representative for Zapstrix, an AI automation company.

Key guidelines:
- Keep responses brief and conversational (1-2 sentences maximum)
- Speak naturally and warmly
- Ask for their name early in the conversation
- Listen carefully and respond appropriately
- Don't interrupt or respond to silence/background noise

About Zapstrix:
- We build custom AI and automation solutions for businesses
- We specialize in lead generation, customer support, and operational efficiency
- We help companies save time and increase revenue through smart automation

Start with: "Hello! Thank you for calling Zapstrix. How can I help you today?"

Remember: Be helpful, not pushy. Quality conversation over sales pressure."""

# Session storage for conversation tracking
ACTIVE_SESSIONS = {}
CALLER_NUMBERS = {}

# Graceful shutdown handling
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Zapstrix Assistant")
    yield
    logger.info("Shutting down Zapstrix Assistant")
    # Cleanup active sessions on shutdown
    for session_id in list(ACTIVE_SESSIONS.keys()):
        try:
            await cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

app = FastAPI(
    title="Zapstrix Assistant",
    description="AI-powered assistant for Zapstrix",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_call_summary(conversation_text: str, caller_name: str = "", caller_number: str = "") -> dict:
    """Extract call information using OpenAI API with proper error handling"""
    if not conversation_text.strip():
        return {
            "caller_name": caller_name or "Unknown",
            "caller_number": caller_number or "Unknown",
            "call_summary": "No conversation content available",
            "call_duration": "Unknown",
            "key_topics": [],
            "follow_up_needed": False
        }
    
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""Analyze this sales call conversation and extract key information.
        
Conversation:
{conversation_text}

Return ONLY valid JSON with these exact keys:
- "caller_name": The caller's name (use "{caller_name}" if provided, otherwise extract from conversation)
- "caller_number": "{caller_number}"
- "call_summary": Brief summary of the call (2-3 sentences)
- "call_duration": Estimate in minutes
- "key_topics": Array of main topics discussed
- "follow_up_needed": Boolean indicating if follow-up is recommended

JSON:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            timeout=10
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Clean up response and extract JSON
        raw_response = re.sub(r"```(?:json)?|```", "", raw_response, flags=re.I).strip()
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        
        if json_match:
            data = json.loads(json_match.group(0))
            
            # Ensure all required fields exist with defaults
            default_data = {
                "caller_name": caller_name or "Unknown",
                "caller_number": caller_number or "Unknown",
                "call_summary": "Call completed successfully",
                "call_duration": "Unknown",
                "key_topics": [],
                "follow_up_needed": False
            }
            
            # Update defaults with extracted data
            for key, value in data.items():
                if key in default_data:
                    default_data[key] = value
            
            return default_data
        else:
            raise ValueError("No valid JSON found in response")
            
    except Exception as e:
        logger.error(f"Error extracting call summary: {e}")
        return {
            "caller_name": caller_name or "Unknown",
            "caller_number": caller_number or "Unknown", 
            "call_summary": f"Call completed - summary extraction failed: {str(e)}",
            "call_duration": "Unknown",
            "key_topics": ["Summary extraction failed"],
            "follow_up_needed": True
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
                "call_type": "inbound_sales"
            }
            
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(N8N_WEBHOOK_URL, json=payload)
                
                if response.status_code < 400:
                    logger.info(f"Successfully sent call data to n8n: {response.status_code}")
                    return True
                else:
                    logger.warning(f"N8N webhook returned status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} to send to n8n failed: {e}")
            if attempt == max_retries - 1:
                logger.error("Failed to send call data to n8n after all retries")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    return False

async def cleanup_session(session_id: str):
    """Clean up session data and send final summary to N8N"""
    if session_id not in ACTIVE_SESSIONS:
        return
    
    session = ACTIVE_SESSIONS[session_id]
    logger.info(f"Cleaning up session {session_id}")
    
    try:
        conversation_text = '\n'.join(session['conversation'])
        
        if conversation_text.strip():
            call_data = extract_call_summary(
                conversation_text, 
                session.get('caller_name', ''),
                session.get('caller_number', 'Unknown')
            )
            
            # Add session metadata
            call_data['session_duration'] = time.time() - session.get('start_time', time.time())
            call_data['total_messages'] = len(session['conversation'])
            
            # Send to N8N
            await send_to_n8n(call_data, session.get('call_sid', 'unknown'))
        
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")
    finally:
        # Always clean up the session
        if session_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[session_id]
        logger.info(f"Session {session_id} cleanup complete")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {
        "message": "Zapstrix Bot - AI-Assistant",
        "version": "1.0.0",
        "status": "operational"
    }

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response"""
    try:
        caller_number = 'Unknown'
        call_sid = None
        
        if request.method == "POST":
            form_data = await request.form()
            caller_number = form_data.get('From', 'Unknown')
            call_sid = form_data.get('CallSid')
        else:
            caller_number = request.query_params.get('From', 'Unknown')
            call_sid = request.query_params.get('CallSid')
        
        logger.info(f"Incoming call from {caller_number}, CallSid: {call_sid}")
        
        # Store caller info
        if call_sid and caller_number != 'Unknown':
            CALLER_NUMBERS[call_sid] = caller_number
        
        # Generate TwiML response
        response = VoiceResponse()
        host = request.url.hostname
        
        connect = Connect()
        stream_url = f'wss://{host}/media-stream'
        connect.stream(url=stream_url)
        response.append(connect)
        
        return HTMLResponse(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        # Return basic TwiML even on error
        response = VoiceResponse()
        response.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
        return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI"""
    session_id = str(uuid.uuid4())
    call_start_time = time.time()
    
    await websocket.accept()
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        # Connect to OpenAI with improved error handling
        logger.info(f"Connecting to OpenAI for session {session_id}")
        
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
            max_size=1024*1024  # 1MB max message size
        ) as openai_ws:
            
            logger.info(f"OpenAI connection established for session {session_id}")
            
            # Initialize session
            ACTIVE_SESSIONS[session_id] = {
                'conversation': [],
                'caller_name': '',
                'caller_number': 'Unknown',
                'call_sid': None,
                'start_time': call_start_time,
                'last_activity': time.time(),
                'openai_ws': openai_ws
            }
            
            # Configure OpenAI session
            await send_session_update(openai_ws)
            
            stream_sid = None
            call_sid = None
            audio_buffer = []
            
            async def send_initial_greeting():
                """Send initial greeting after connection established"""
                await asyncio.sleep(2)  # Wait for connection to stabilize
                if openai_ws.open and session_id in ACTIVE_SESSIONS:
                    try:
                        # Send greeting message
                        greeting_message = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Please start the call with your greeting message."}]
                            }
                        }
                        await openai_ws.send(json.dumps(greeting_message))
                        await asyncio.sleep(0.1)
                        
                        # Trigger response
                        await openai_ws.send(json.dumps({"type": "response.create"}))
                        logger.info(f"Sent initial greeting for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error sending initial greeting: {e}")
            
            def is_meaningful_audio(transcript: str) -> bool:
                """Filter out noise, background sounds, and unclear speech"""
                if not transcript or len(transcript.strip()) < 2:
                    return False
                
                # Filter out common noise patterns
                noise_patterns = [
                    r'^[^a-zA-Z]*$',  # Only punctuation/symbols
                    r'^(uh|um|ah|hm|mm|hmm)+$',  # Only filler sounds
                    r'^[.,!?;:\s]+$',  # Only punctuation and spaces
                    r'^\[.*\]$',  # Transcription annotations
                    r'^<.*>$',  # XML-like tags
                    r'^(you|thank|thanks|okay|ok|yes|yeah|no|hi|hello)$',  # Very short responses
                ]
                
                transcript_clean = transcript.lower().strip()
                
                for pattern in noise_patterns:
                    if re.match(pattern, transcript_clean):
                        return False
                
                # Must contain at least one clear word
                words = re.findall(r'\b[a-zA-Z]{2,}\b', transcript_clean)
                return len(words) > 0
            
            async def receive_from_twilio():
                """Receive audio data from Twilio"""
                nonlocal stream_sid, call_sid
                
                try:
                    async for message in websocket.iter_text():
                        if shutdown_event.is_set():
                            break
                            
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        
                        if data['event'] == 'media' and openai_ws.open:
                            # Forward audio to OpenAI
                            try:
                                payload = data['media']['payload']
                                if payload and len(payload) > 0:
                                    audio_append = {
                                        "type": "input_audio_buffer.append",
                                        "audio": payload
                                    }
                                    await openai_ws.send(json.dumps(audio_append))
                            except (KeyError, TypeError, websockets.ConnectionClosed):
                                continue
                            
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            call_sid = data['start'].get('callSid')
                            
                            logger.info(f"Stream started - StreamSid: {stream_sid}, CallSid: {call_sid}")
                            
                            # Get caller number
                            caller_number = CALLER_NUMBERS.get(call_sid, 'Unknown')
                            
                            # Update session
                            if session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id].update({
                                    'call_sid': call_sid,
                                    'caller_number': caller_number
                                })
                            
                            # Send initial greeting
                            asyncio.create_task(send_initial_greeting())
                            
                        elif data['event'] == 'stop':
                            logger.info(f"Twilio stream stopped for session {session_id}")
                            break
                            
                except WebSocketDisconnect:
                    logger.info(f"Twilio WebSocket disconnected for session {session_id}")
                except Exception as e:
                    logger.error(f"Error in receive_from_twilio: {e}")
                finally:
                    if call_sid and call_sid in CALLER_NUMBERS:
                        del CALLER_NUMBERS[call_sid]
            
            async def send_to_twilio():
                """Send audio responses back to Twilio"""
                nonlocal stream_sid
                
                try:
                    async for openai_message in openai_ws:
                        if shutdown_event.is_set():
                            break
                            
                        try:
                            response = json.loads(openai_message)
                        except json.JSONDecodeError:
                            continue
                        
                        # Update last activity
                        if session_id in ACTIVE_SESSIONS:
                            ACTIVE_SESSIONS[session_id]['last_activity'] = time.time()
                        
                        # Handle different response types
                        if response['type'] == 'response.audio_transcript.done':
                            transcript = response.get('transcript', '')
                            if transcript and session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id]['conversation'].append(f"Assistant: {transcript}")
                                logger.info(f"Assistant said: {transcript}")
                        
                        elif response['type'] == 'conversation.item.input_audio_transcription.completed':
                            transcript = response.get('transcript', '')
                            
                            # Only process meaningful audio
                            if transcript and is_meaningful_audio(transcript) and session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id]['conversation'].append(f"Caller: {transcript}")
                                logger.info(f"Caller said: {transcript}")
                                
                                # Extract caller name if not already captured
                                if not ACTIVE_SESSIONS[session_id]['caller_name']:
                                    # Simple name extraction
                                    name_patterns = [
                                        r"(?:i'm|i am|my name is|this is|it's)\s+([a-zA-Z]+)",
                                        r"^([a-zA-Z]+)(?:\s+speaking)?$"
                                    ]
                                    for pattern in name_patterns:
                                        match = re.search(pattern, transcript.lower().strip())
                                        if match:
                                            potential_name = match.group(1).title()
                                            if len(potential_name) > 1 and potential_name.isalpha():
                                                ACTIVE_SESSIONS[session_id]['caller_name'] = potential_name
                                                logger.info(f"Extracted caller name: {potential_name}")
                                                break
                        
                        # Send audio back to Twilio
                        elif response['type'] == 'response.audio.delta' and response.get('delta'):
                            if stream_sid and websocket.client_state.name == 'CONNECTED':
                                try:
                                    audio_payload = response['delta']
                                    if audio_payload:
                                        audio_delta = {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": audio_payload}
                                        }
                                        await websocket.send_json(audio_delta)
                                except Exception as e:
                                    logger.error(f"Error sending audio to Twilio: {e}")
                        
                        # Handle session errors
                        elif response['type'] == 'error':
                            logger.error(f"OpenAI error: {response}")
                            
                except websockets.ConnectionClosed:
                    logger.info(f"OpenAI WebSocket closed for session {session_id}")
                except Exception as e:
                    logger.error(f"Error in send_to_twilio: {e}")
            
            # Run both tasks concurrently
            try:
                await asyncio.gather(
                    receive_from_twilio(),
                    send_to_twilio(),
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"Error in WebSocket handling: {e}")
                
    except Exception as e:
        logger.error(f"Error establishing OpenAI connection: {e}")
    finally:
        # Cleanup session
        if session_id in ACTIVE_SESSIONS:
            await cleanup_session(session_id)

async def send_session_update(openai_ws):
    """Configure OpenAI session parameters"""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw", 
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.6,
            "input_audio_transcription": {
                "model": "whisper-1"
            }
        }
    }
    
    try:
        await openai_ws.send(json.dumps(session_update))
        logger.info("Session configuration sent to OpenAI")
        await asyncio.sleep(0.5)  # Wait for configuration to take effect
    except Exception as e:
        logger.error(f"Error configuring OpenAI session: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "n8n_configured": bool(N8N_WEBHOOK_URL),
        "uptime": time.time(),
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "type": type(exc).__name__}
    )

if __name__ == "__main__":
    import uvicorn
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=PORT,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)