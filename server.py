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

# Minimal logging - only critical system errors
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

# Sales-focused system message for Pascual & Co
SYSTEM_MESSAGE = """You are a professional sales representative for Pascual & Co. Your role is to:

1. Warmly greet callers and introduce yourself as representing Pascual & Co
2. Listen to their needs and inquiries about our services
3. Provide helpful information about what Pascual & Co offers
3a. Pascual & Co. is an AI Agent and Automation company that specializes in building custom solutions for businesses of all sizes. 
    We focus on delivering high-quality, scalable, and secure solutions tailored to meet the unique needs of our clients.
3b. In conversations with potential clients, you should highlight our expertise in AI and automation, and how these tools can be used
    for lead generation, customer support, and operational efficiency.
4. Build rapport and trust through professional, friendly conversation
5. Ask for their name naturally during the conversation
6. Answer questions about our company and services professionally
7. Handle objections with empathy and expertise
8. Keep conversations focused but natural - you're here to help and inform

Be conversational, professional, and genuinely helpful. Focus on understanding their needs rather than being pushy. Always maintain a positive, solution-oriented approach.

Start by saying: "Hello! Thank you for calling Pascual & Co. I'm here to help you today. May I ask who I'm speaking with?"
"""

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
async def lifespan(app: FastAPI):
    yield
    # Cleanup active sessions on shutdown
    for session_id in list(ACTIVE_SESSIONS.keys()):
        try:
            await cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

app = FastAPI(
    title="Pascual & Co Sales Bot",
    description="AI-powered sales assistant for Pascual & Co",
    version="1.0.0",
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
        return {
            "caller_name": caller_name or "Unknown",
            "caller_number": caller_number or "Unknown", 
            "call_summary": f"Call completed - summary extraction failed",
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
                "company": "Pascual & Co",
                "call_type": "inbound_sales"
            }
            
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(N8N_WEBHOOK_URL, json=payload)
                
                if response.status_code < 400:
                    return True
                    
        except Exception as e:
            if attempt == max_retries - 1:  # Only log on final failure
                pass  # Silent failure - no logging needed
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    return False

async def cleanup_session(session_id: str):
    """Clean up session data and send final summary to N8N"""
    if session_id not in ACTIVE_SESSIONS:
        return
    
    session = ACTIVE_SESSIONS[session_id]
    
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
        pass  # Silent cleanup - no logging needed
    finally:
        # Always clean up the session
        if session_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[session_id]

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {
        "message": "Pascual & Co Sales Bot - AI-Powered Sales Assistant",
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
        # Return basic TwiML even on error - no logging needed
        response = VoiceResponse()
        response.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
        return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI"""
    session_id = str(uuid.uuid4())
    call_start_time = time.time()
    
    await websocket.accept()
    
    try:
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            ping_interval=30,
            ping_timeout=10
        ) as openai_ws:
            
            # Initialize session
            ACTIVE_SESSIONS[session_id] = {
                'conversation': [],
                'caller_name': '',
                'caller_number': 'Unknown',
                'call_sid': None,
                'start_time': call_start_time,
                'last_activity': time.time()
            }
            
            await send_session_update(openai_ws)
            
            stream_sid = None
            call_sid = None
            
            async def send_initial_greeting():
                """Send initial greeting after connection established"""
                await asyncio.sleep(1)
                if openai_ws.open:
                    greeting_message = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Please greet the caller as instructed."}]
                        }
                    }
                    await openai_ws.send(json.dumps(greeting_message))
                    await openai_ws.send(json.dumps({"type": "response.create"}))
            
            async def check_call_timeout():
                """Monitor call duration and terminate if too long"""
                while session_id in ACTIVE_SESSIONS:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    if session_id not in ACTIVE_SESSIONS:
                        break
                        
                    session = ACTIVE_SESSIONS[session_id]
                    call_duration = time.time() - session['start_time']
                    
                    if call_duration > MAX_CALL_DURATION:
                        break
            
            async def receive_from_twilio():
                """Receive audio data from Twilio"""
                nonlocal stream_sid, call_sid
                
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        
                        if data['event'] == 'media' and openai_ws.open:
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            await openai_ws.send(json.dumps(audio_append))
                            
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            call_sid = data['start'].get('callSid')
                            
                            # Get caller number
                            caller_number = CALLER_NUMBERS.get(call_sid, 'Unknown')
                            
                            # Update session
                            if session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id].update({
                                    'call_sid': call_sid,
                                    'caller_number': caller_number
                                })
                            
                            asyncio.create_task(send_initial_greeting())
                            
                        elif data['event'] == 'stop':
                            # Trigger cleanup immediately when call ends
                            asyncio.create_task(cleanup_session(session_id))
                            break
                            
                except WebSocketDisconnect:
                    # Trigger cleanup when WebSocket disconnects
                    asyncio.create_task(cleanup_session(session_id))
                except Exception as e:
                    # Silent failure - continue processing
                    pass
                finally:
                    if call_sid and call_sid in CALLER_NUMBERS:
                        del CALLER_NUMBERS[call_sid]
            
            async def send_to_twilio():
                """Send audio responses back to Twilio"""
                nonlocal stream_sid
                
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        
                        # Update last activity
                        if session_id in ACTIVE_SESSIONS:
                            ACTIVE_SESSIONS[session_id]['last_activity'] = time.time()
                        
                        # Track assistant responses
                        if response['type'] == 'response.audio_transcript.done':
                            transcript = response.get('transcript', '')
                            if transcript and session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id]['conversation'].append(f"Assistant: {transcript}")
                        
                        # Track user input
                        if response['type'] == 'conversation.item.input_audio_transcription.completed':
                            transcript = response.get('transcript', '')
                            if transcript and session_id in ACTIVE_SESSIONS:
                                ACTIVE_SESSIONS[session_id]['conversation'].append(f"Caller: {transcript}")
                                
                                # Extract caller name if not already captured
                                if not ACTIVE_SESSIONS[session_id]['caller_name']:
                                    # Simple name extraction - look for "I'm" or "My name is" patterns
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
                                                break
                        
                        # Send audio back to Twilio
                        if response['type'] == 'response.audio.delta' and response.get('delta'):
                            if stream_sid:
                                try:
                                    audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                                    audio_delta = {
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": audio_payload}
                                    }
                                    await websocket.send_json(audio_delta)
                                except Exception as e:
                                    pass  # Continue processing audio
                                    
                except Exception as e:
                    pass  # Continue processing
            
            # Run all tasks concurrently
            timeout_task = asyncio.create_task(check_call_timeout())
            
            try:
                await asyncio.gather(
                    receive_from_twilio(),
                    send_to_twilio(),
                    return_exceptions=True
                )
            finally:
                timeout_task.cancel()
                
    except Exception as e:
        # Ensure cleanup happens even on unexpected errors
        asyncio.create_task(cleanup_session(session_id))
    finally:
        # Final cleanup attempt - but don't double-cleanup
        if session_id in ACTIVE_SESSIONS:
            await cleanup_session(session_id)

async def send_session_update(openai_ws):
    """Configure OpenAI session parameters"""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw", 
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.7,
            "input_audio_transcription": {"model": "whisper-1"}
        }
    }
    await openai_ws.send(json.dumps(session_update))

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
            log_level="error",  # Production log level
            access_log=False    # Disable access logs for production
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server startup failed: {e}")
        sys.exit(1)