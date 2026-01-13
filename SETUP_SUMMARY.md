# Setup Summary - Gabriel's Voice Agent

Quick summary of all setup steps to get your personal voice agent running with Google Sheets logging.

## What You've Got

✅ **Personalized Voice Agent** - AI assistant with your professional background
✅ **Google Sheets Logging** - All calls automatically logged to a Google Sheet
✅ **Cost-Efficient** - Uses gpt-4o-mini-realtime for 60% cost savings
✅ **Complete Documentation** - Step-by-step guides for everything

## Setup Order

Follow these steps in order:

### 1. Google Sheets Setup (15-20 min)
**Why First**: Get credentials ready before deployment

**Steps**:
1. Create Google Cloud Project
2. Enable Google Sheets API and Google Drive API
3. Create service account
4. Download JSON credentials
5. See: [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md)

**Output**: JSON credentials file

### 2. Get OpenAI API Key (5 min)
1. Go to https://platform.openai.com
2. Create account / Add payment method
3. Generate API key
4. Set spending limit ($50/month recommended)

**Output**: API key starting with `sk-proj-...`

### 3. Get Twilio Phone Number (10 min)
1. Sign up at https://www.twilio.com/try-twilio
2. Purchase a phone number (~$1-2/month)
3. Choose a Miami number (305/786) recommended

**Output**: Twilio phone number

### 4. Deploy to Render (10-15 min)
**See**: [DEPLOYMENT.md](DEPLOYMENT.md)

**Steps**:
1. Push code to GitHub
2. Connect repository to Render
3. Configure build/start commands
4. Add environment variables:
   ```
   OPENAI_API_KEY=your_key_here
   GOOGLE_SHEETS_CREDS_JSON=paste_entire_json_here
   GOOGLE_SHEET_NAME=Voice Agent Call Logs
   PORT=10000
   VOICE=sage
   TEMPERATURE=0.8
   LOG_LEVEL=INFO
   ```
5. Deploy!

**Output**: Deployment URL (e.g., `https://gabriel-voice-agent.onrender.com`)

### 5. Configure Twilio Webhook (5 min)
**See**: [TWILIO_SETUP.md](TWILIO_SETUP.md)

**Steps**:
1. Go to Twilio Console → Phone Numbers
2. Click your phone number
3. Under "A CALL COMES IN":
   - Webhook: `https://your-deployment-url.com/incoming-call`
   - Method: POST
4. Save

**Output**: Phone number ready to receive calls

### 6. Test Everything (10 min)
1. Call your Twilio number
2. Have a conversation
3. End the call
4. Check your Google Sheet at https://sheets.google.com
5. Verify log entry appears

**Output**: Working voice agent!

### 7. Custom Domain (Optional, 15 min)
**See**: [DOMAIN_SETUP.md](DOMAIN_SETUP.md)

**Steps**:
1. Add `voice.gabepascual.com` in Render
2. Add CNAME record in DNS
3. Wait for SSL certificate
4. Update Twilio webhook to use custom domain

**Output**: Professional domain for your agent

## Environment Variables Checklist

Make sure all these are set in your deployment platform:

- [ ] `OPENAI_API_KEY` - From OpenAI platform
- [ ] `GOOGLE_SHEETS_CREDS_JSON` - Entire JSON from service account
- [ ] `GOOGLE_SHEET_NAME` - "Voice Agent Call Logs" (or custom name)
- [ ] `PORT` - 10000 (for Render) or 8080 (for Fly.io)
- [ ] `VOICE` - sage (or alloy/echo/nova/etc.)
- [ ] `TEMPERATURE` - 0.8
- [ ] `LOG_LEVEL` - INFO

## Testing Checklist

Before going live:

- [ ] Health endpoint works: `https://your-url.com/health`
- [ ] Call connects successfully
- [ ] AI greets you professionally
- [ ] Can understand your speech
- [ ] AI responds appropriately
- [ ] Call ends cleanly
- [ ] Log appears in Google Sheets
- [ ] Log includes timestamp, phone number, duration, summary
- [ ] No errors in deployment logs

## Costs Breakdown

### One-Time Setup
- **Google Cloud**: $0 (free tier)
- **Time**: ~1-2 hours

### Monthly Recurring (Light Use - 50 min/month)
- **OpenAI API**: ~$0.50
- **Twilio Number**: ~$1-2
- **Twilio Usage**: ~$0.43
- **Google Sheets**: $0 (free)
- **Deployment (Render)**: $0 (free tier) or $7 (paid)
- **Total**: $2-10/month

### Per Call
- **OpenAI**: ~$0.60/hour
- **Twilio**: ~$0.51/hour
- **Total**: ~$1.11/hour of conversation

## Quick Links

### Setup Guides
- [Google Sheets Setup](GOOGLE_SHEETS_SETUP.md) - Configure Google Sheets API
- [Deployment Guide](DEPLOYMENT.md) - Deploy to Render/Railway/Fly.io
- [Twilio Setup](TWILIO_SETUP.md) - Configure phone number
- [Domain Setup](DOMAIN_SETUP.md) - Custom domain (optional)

### After Setup
- [Quick Reference](QUICK_REFERENCE.md) - Common commands and URLs
- [README](README.md) - Full project documentation

### External Services
- [OpenAI Platform](https://platform.openai.com)
- [Twilio Console](https://console.twilio.com)
- [Google Cloud Console](https://console.cloud.google.com)
- [Google Sheets](https://sheets.google.com)
- [Render Dashboard](https://dashboard.render.com)

## Troubleshooting Quick Fixes

### "GOOGLE_SHEETS_CREDS_JSON not set"
- Check environment variable is set in Render
- Paste entire JSON (all quotes and special characters)
- Redeploy after adding variable

### "Permission denied" (Google Sheets)
- Verify Google Sheets API is enabled
- Verify Google Drive API is enabled
- Let app create sheet automatically (don't create manually)

### Call connects but no audio
- Check OpenAI API key is correct
- Verify Realtime API access (not just standard API)
- Check OpenAI account has credits

### Twilio webhook fails
- Ensure URL is HTTPS (not HTTP)
- URL must end with `/incoming-call`
- Test health endpoint first: `https://your-url.com/health`

### Sheet not updating
- Check deployment logs for errors
- Verify service account has Editor access (not Viewer)
- Try redeploying

## What to Expect

### First Call Experience
1. You dial your Twilio number
2. AI answers: "Hi, I'm your personal AI voice assistant..."
3. AI asks how it can help
4. You have a natural conversation
5. AI responds appropriately based on your resume/background
6. You end the call
7. Check Google Sheets - new row appears with call details!

### Google Sheet Structure
| Timestamp | Call SID | From | To | Event | Duration | Exchanges | Summary |
|-----------|----------|------|----|----|---------|-----------|---------|
| 2026-01-12 15:30 | CA123... | +1786... | +1555... | call_ended | 125 | 10 | User asked about availability... |

## Next Steps After Setup

1. **Test extensively** - Make 10-20 test calls
2. **Monitor costs** - Check OpenAI and Twilio usage daily
3. **Fine-tune prompt** - Adjust based on call experiences
4. **Try different voices** - Test alloy, nova, sage
5. **Set spending limits** - In OpenAI and Twilio dashboards
6. **Share selectively** - Test with friends before wider use

## Support

If you get stuck:
1. Check the specific guide for your issue
2. Review deployment logs
3. Test each component individually
4. Check service status pages (OpenAI, Twilio, Render)

## Success Criteria

You're ready when:
- ✅ Can call your number anytime
- ✅ AI answers and sounds natural
- ✅ Logs appear in Google Sheets automatically
- ✅ Costs are within expected range
- ✅ No errors in deployment logs
- ✅ Custom domain working (if you set it up)

---

**Total Setup Time**: ~1-2 hours
**Monthly Cost**: $2-10 for light use
**Ongoing Maintenance**: ~15 min/week (check logs and costs)

Ready to start? Begin with [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md)! 🚀
