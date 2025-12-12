import os
import json

from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from openai import OpenAI

# ============================================================
#  Load environment & config
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # or gpt-4.1-mini etc.

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required in your .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Store conversation history per call
conversation_history = {}

# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI()

# Optional CORS (mostly irrelevant for Twilio, but harmless)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  Elmo "off-track" classifier using OpenAI
# ============================================================

def analyze_meeting_snippet(text: str, full_conversation: str = None) -> dict:
    """
    Send the caller's speech to OpenAI and decide:
    - Is the meeting off-track / redundant?
    - What should Elmo say?
    Returns dict like:
      {
        "off_topic": true/false,
        "moderator_message": "Everyone, let's move on..."
      }
    """

    system_prompt = (
        "You are Elmo, a meeting facilitator who ONLY interrupts when absolutely necessary.\n\n"
        "Your job is to analyze meeting conversations and decide if intervention is needed.\n"
        "You must respond ONLY with JSON like:\n"
        "{\n"
        '  \"off_topic\": true or false,\n'
        '  \"moderator_message\": \"short sentence Elmo should say to the group\"\n'
        "}\n\n"
        "IMPORTANT RULES:\n"
        "- off_topic = true ONLY when the conversation is clearly:\n"
        "  * Going in circles for multiple exchanges\n"
        "  * Completely unrelated to work/meeting topics\n"
        "  * Stuck in an unproductive argument\n"
        "  * Excessively repeating the same point\n\n"
        "- off_topic = false when:\n"
        "  * Normal discussion is happening\n"
        "  * People are asking questions or clarifying\n"
        "  * The topic is work-related or on-task\n"
        "  * There's natural back-and-forth dialogue\n"
        "  * People are making progress, even slowly\n\n"
        "- moderator_message should be SHORT (5-10 words max), friendly, and helpful.\n"
        "- When in doubt, set off_topic to FALSE. It's better to stay quiet than interrupt unnecessarily."
    )

    # Include conversation history if available for better context
    if full_conversation:
        user_prompt = (
            f"Here is the recent conversation history:\n\n{full_conversation}\n\n"
            "Based on this conversation, is Elmo's intervention needed? Respond in JSON only."
        )
    else:
        user_prompt = (
            f"Here is the latest snippet of conversation:\n\n{text}\n\n"
            "Based on this, is Elmo's intervention needed? Respond in JSON only."
        )

    # Using Chat Completions in JSON mode
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if something weird happens
        data = {
            "off_topic": True,
            "moderator_message": "Everyone, let's move on to the next topic.",
        }

    # Ensure keys exist
    if "off_topic" not in data:
        data["off_topic"] = True
    if "moderator_message" not in data:
        data["moderator_message"] = "Everyone, let's move on to the next topic."

    return data


# ============================================================
#  Basic health check
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Elmo voice agent (non-realtime) is running.",
        "active_calls": len(conversation_history)
    }


@app.post("/cleanup")
async def cleanup(CallSid: str = Form(default=None)):
    """
    Optional endpoint for Twilio to call when a call ends
    to clean up conversation history
    """
    if CallSid and CallSid in conversation_history:
        del conversation_history[CallSid]
        print(f"[Elmo] Cleaned up conversation history for {CallSid}")
    return {"status": "cleaned"}


# ============================================================
#  Twilio Voice Webhook: /voice
#
#  - First call: no SpeechResult -> greet & start Gather
#  - Subsequent calls: SpeechResult present -> analyze & respond
# ============================================================

@app.post("/voice", response_class=PlainTextResponse)
async def voice(
    SpeechResult: str = Form(default=None),
    From: str = Form(default=None),
    To: str = Form(default=None),
    CallSid: str = Form(default=None),
):
    """
    Twilio sends:
    - initial request with no SpeechResult
    - follow-up request with SpeechResult (transcription of user's speech)
    We respond with TwiML.
    """

    # First hit: no speech yet -> introduce Elmo & start listening
    if not SpeechResult:
        # Initialize conversation history for this call
        conversation_history[CallSid] = []
        twiml = """
<Response>
    <Say voice="Polly.Joanna">
        Hi, I'm Elmo, your meeting co-pilot.
        I'll listen quietly and only jump in if things get stuck or off track.
    </Say>
    <Gather input="speech"
            action="/voice"
            method="POST"
            speechTimeout="auto"
            speechModel="experimental_conversations">
    </Gather>
</Response>
""".strip()
        return PlainTextResponse(content=twiml, media_type="application/xml")

    # We have a SpeechResult from Twilio's speech-to-text
    snippet = SpeechResult.strip()

    print(f"[Twilio] CallSid={CallSid}  From={From}  To={To}")
    print(f"[Twilio] SpeechResult: {snippet}")

    # Skip very short snippets (likely partial captures or noise)
    word_count = len(snippet.split())
    if word_count < 5:
        print(f"[Elmo] Skipping short snippet ({word_count} words) - continuing to listen")
        twiml = """
<Response>
    <Gather input="speech"
            action="/voice"
            method="POST"
            speechTimeout="auto"
            speechModel="experimental_conversations">
    </Gather>
</Response>
""".strip()
        return PlainTextResponse(content=twiml, media_type="application/xml")

    # Add to conversation history
    if CallSid not in conversation_history:
        conversation_history[CallSid] = []

    conversation_history[CallSid].append(snippet)

    # Keep only last 10 exchanges to avoid token overload
    if len(conversation_history[CallSid]) > 10:
        conversation_history[CallSid] = conversation_history[CallSid][-10:]

    # Build full conversation context
    full_conversation = "\n".join(conversation_history[CallSid])
    total_word_count = len(full_conversation.split())

    # Only analyze after we have enough content (at least 30 words)
    if total_word_count < 30:
        print(f"[Elmo] Not enough context yet ({total_word_count} words) - continuing to listen")
        twiml = """
<Response>
    <Gather input="speech"
            action="/voice"
            method="POST"
            speechTimeout="auto"
            speechModel="experimental_conversations">
    </Gather>
</Response>
""".strip()
        return PlainTextResponse(content=twiml, media_type="application/xml")

    # Analyze with full conversation context
    print(f"[Elmo] Analyzing conversation ({total_word_count} words)...")
    analysis = analyze_meeting_snippet(snippet, full_conversation)
    off_topic = bool(analysis.get("off_topic", False))
    moderator_message = analysis.get(
        "moderator_message", "Let's refocus on the main topic."
    )

    print(f"[Elmo] Analysis: off_topic={off_topic}, message={moderator_message}")

    if off_topic:
        # Meeting is off-track -> Elmo interrupts
        print("[Elmo] INTERRUPTING - conversation is off track")

        # Clear conversation history after intervention to allow fresh start
        # This prevents the AI from continuously seeing old off-topic context
        conversation_history[CallSid] = []
        print("[Elmo] Cleared conversation history to allow fresh analysis after intervention")

        twiml = f"""
<Response>
    <Say voice="Polly.Joanna">
        {moderator_message}
    </Say>
    <Gather input="speech"
            action="/voice"
            method="POST"
            speechTimeout="auto"
            speechModel="experimental_conversations">
    </Gather>
</Response>
""".strip()
    else:
        # Meeting is fine -> stay silent and keep listening
        print("[Elmo] Staying quiet - conversation is on track")
        twiml = """
<Response>
    <Gather input="speech"
            action="/voice"
            method="POST"
            speechTimeout="auto"
            speechModel="experimental_conversations">
    </Gather>
</Response>
""".strip()

    return PlainTextResponse(content=twiml, media_type="application/xml")


# ============================================================
#  Run with:  uvicorn test:app --host 0.0.0.0 --port 5024
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "5024"))
    uvicorn.run("test:app", host="0.0.0.0", port=port, reload=True)
