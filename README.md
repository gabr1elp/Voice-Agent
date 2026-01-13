# Gabriel's Personal Voice Agent

A professional AI voice assistant powered by OpenAI's Realtime API (gpt-4o-mini-realtime) and Twilio. This agent acts as Gabriel Pascual's personal representative, handling incoming calls, screening inquiries, taking messages, and providing information about Gabriel's background and availability.

## Features

- **Real-time Voice Conversation**: Natural conversation using OpenAI's Realtime API with low latency
- **Professional Screening**: Intelligently handles inquiries about Gabriel's work, experience, and availability
- **Google Sheets Logging**: Automatically logs all calls to Google Sheets for easy access from anywhere
- **Message Taking**: Captures caller details, purpose of call, and callback information
- **Cost-Efficient**: Uses gpt-4o-mini-realtime for 60% cost savings compared to standard GPT-4o-realtime
- **Free Logging**: No paid database services - uses free Google Sheets API

## Tech Stack

- **Backend**: Python, FastAPI
- **AI**: OpenAI Realtime API (gpt-4o-mini-realtime-preview)
- **Telephony**: Twilio Voice with Media Streams
- **Logging**: Google Sheets API (gspread)
- **WebSockets**: Real-time bidirectional audio streaming
- **Deployment**: Render / Railway / Fly.io (your choice)

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key with Realtime API access
- Twilio account with a phone number
- Google Cloud account (free) for Sheets API
- Deployment platform account (Render recommended)

### Setup Steps

1. **Follow the Setup Guide**: See [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) for step-by-step instructions

2. **Configure Google Sheets**: See [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md) for logging setup

3. **Deploy**: See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions

4. **Configure Twilio**: See [TWILIO_SETUP.md](TWILIO_SETUP.md) for phone integration

5. **Custom Domain** (Optional): See [DOMAIN_SETUP.md](DOMAIN_SETUP.md) for voice.gabepascual.com

## Documentation

| Guide | What It Covers |
|-------|----------------|
| [README.md](README.md) | Overview and quick start (this file) |
| [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) | Complete step-by-step setup checklist |
| [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md) | Google Sheets API configuration for call logging |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deploy to Render/Railway/Fly.io |
| [TWILIO_SETUP.md](TWILIO_SETUP.md) | Twilio phone number and webhook setup |
| [DOMAIN_SETUP.md](DOMAIN_SETUP.md) | Custom domain configuration |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick commands and URLs |

## Project Structure

```
Voice-Agent/
├── realtime_mini.py              # Main application
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not committed)
├── .gitignore                   # Git ignore rules
├── README.md                     # This file
├── SETUP_CHECKLIST.md           # Complete setup guide
├── GOOGLE_SHEETS_SETUP.md       # Google Sheets integration guide
├── DEPLOYMENT.md                 # Deployment instructions
├── TWILIO_SETUP.md              # Twilio configuration
├── DOMAIN_SETUP.md              # Custom domain setup
└── QUICK_REFERENCE.md           # Quick reference guide
```

## How It Works

### Call Flow

1. **Incoming Call** → Twilio receives call on your phone number
2. **Webhook** → Twilio sends webhook to `/incoming-call` endpoint
3. **TwiML Response** → Server returns TwiML opening a WebSocket Media Stream
4. **WebSocket Bridge** → Server connects to both Twilio and OpenAI
5. **Audio Relay** → Audio flows bidirectionally between caller and AI
6. **Call Logging** → Transcripts logged to Google Sheets in real-time
7. **Call End** → Summary generated and saved to Google Sheets

### Call Logging

All calls are automatically logged to a Google Sheet with:
- **Timestamp**: When the call occurred
- **Call SID**: Unique Twilio identifier
- **From/To Numbers**: Caller and recipient phone numbers
- **Event Type**: call_started or call_ended
- **Duration**: Call length in seconds
- **Exchanges**: Number of conversation turns
- **Summary**: AI-generated conversation summary

Access your call logs anytime at: https://sheets.google.com

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | Your OpenAI API key with Realtime access |
| GOOGLE_SHEETS_CREDS_JSON | Yes | - | Google service account credentials (JSON string) |
| GOOGLE_SHEET_NAME | No | "Voice Agent Call Logs" | Name of your Google Sheet |
| PORT | No | 5025 | Port to run the server on |
| VOICE | No | sage | OpenAI voice (alloy, echo, fable, onyx, nova, shimmer, sage) |
| TEMPERATURE | No | 0.8 | Model temperature (0.0-1.0) |
| LOG_LEVEL | No | DEBUG | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Voice Options

Available OpenAI Realtime voices:
- **sage** (default) - Professional, warm
- **alloy** - Neutral, balanced
- **echo** - Warm, friendly
- **fable** - British accent, storytelling
- **onyx** - Deep, authoritative
- **nova** - Energetic, bright
- **shimmer** - Soft, calm

## API Endpoints

### `GET /`
Health check and service info.

### `GET /health`
Detailed health status with active session count.

### `POST /incoming-call`
Twilio webhook for incoming calls. Returns TwiML.

### `WebSocket /media-stream`
WebSocket endpoint for Twilio Media Streams. Handles bidirectional audio.

## Cost Estimate

### Development/Testing
- **OpenAI**: ~$0.60/hour of conversation
- **Twilio**: $0.0085/minute (~$0.51/hour)
- **Google Sheets API**: Free
- **Platform**: Free tier (Render/Railway/Fly.io)
- **Total**: ~$1.11/hour of calls

### Production (Light Use)
Assuming 50 minutes of calls per month:
- **OpenAI**: ~$0.50
- **Twilio**: $1-2 (phone number) + $0.43 (calls) = ~$2.43/month
- **Google Sheets**: $0 (free tier)
- **Platform**: Free tier or $5-7/month
- **Total**: $3-10/month

## Security & Privacy

- **No secrets in code**: All API keys via environment variables
- **HTTPS only**: Twilio requires HTTPS (auto-configured on platforms)
- **Secure logging**: Google Sheets access via service account
- **No audio recordings**: Conversations are transcribed but not audio-recorded
- **Private by default**: Only you have access to the Google Sheet

## Customization

### Update System Prompt

Edit the AI's behavior in `realtime_mini.py`:

```python
VOICE_SYSTEM_PROMPT = """
You are Gabriel Pascual's personal AI voice assistant.
[Customize this section to your needs]
"""
```

### Change Voice

Update `.env`:
```env
VOICE=nova
```

### Custom Sheet Name

Update `.env`:
```env
GOOGLE_SHEET_NAME=My Call Logs
```

## Testing

### Test Scenarios

- [ ] AI greets caller appropriately
- [ ] AI can answer questions about Gabriel's background
- [ ] AI can take messages (name, number, purpose)
- [ ] AI handles unclear speech gracefully
- [ ] Call log appears in Google Sheets after call ends
- [ ] Summary includes key conversation points
- [ ] Duration is accurately recorded

## Troubleshooting

See individual guides for detailed troubleshooting:
- [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md#troubleshooting) - Sheets API issues
- [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting) - Deployment issues
- [TWILIO_SETUP.md](TWILIO_SETUP.md#troubleshooting) - Call connection issues

## Future Enhancements

Possible improvements:
- [ ] SMS notifications when calls come in
- [ ] Voicemail transcription
- [ ] Multi-language support
- [ ] Calendar integration for scheduling
- [ ] CRM integration (HubSpot, Salesforce)
- [ ] Call recording option
- [ ] Analytics dashboard with charts

## Support & Documentation

- **Google Sheets Setup**: [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md)
- **Full Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Twilio Setup**: [TWILIO_SETUP.md](TWILIO_SETUP.md)
- **Domain Configuration**: [DOMAIN_SETUP.md](DOMAIN_SETUP.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **OpenAI Realtime API Docs**: https://platform.openai.com/docs/guides/realtime
- **Twilio Media Streams Docs**: https://www.twilio.com/docs/voice/media-streams
- **Google Sheets API Docs**: https://developers.google.com/sheets/api

## About

Built by Gabriel Pascual, Data/AI Analyst at Accenture, specializing in AI-driven solutions and automation. This voice agent showcases hierarchical agent architecture, real-time AI processing, and practical deployment of conversational AI.

**Contact**:
- Email: pascualgabriel0423@gmail.com
- Phone: (786) 253-4432

---

**License**: Private/Personal Use
