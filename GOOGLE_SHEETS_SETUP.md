# Google Sheets Setup Guide

This guide walks you through setting up Google Sheets integration for call logging. All your call logs will be automatically saved to a Google Sheet that you can access from anywhere.

## Why Google Sheets?

Instead of storing logs on the server (which you can't easily access), Google Sheets provides:
- **Easy Access**: View logs from any device, anywhere
- **Real-time Updates**: Logs appear immediately after each call
- **Free**: No cost for Google Sheets API usage
- **Analysis Ready**: Use Sheets formulas, charts, and filters
- **Shareable**: Share with team members or assistants if needed

## Prerequisites

- Google account
- 15-20 minutes for setup

---

## Step 1: Create a Google Cloud Project

### 1.1: Go to Google Cloud Console

Visit: https://console.cloud.google.com

### 1.2: Create New Project

1. Click the project dropdown at the top
2. Click "New Project"
3. Name: `Voice Agent Logging` (or your preferred name)
4. Click "Create"
5. Wait for project creation (usually < 1 minute)
6. Select your new project from the dropdown

---

## Step 2: Enable Google Sheets API

### 2.1: Enable the API

1. In the left sidebar, go to "APIs & Services" → "Library"
2. Search for "Google Sheets API"
3. Click on it
4. Click "Enable"
5. Wait for it to enable (< 30 seconds)

### 2.2: Enable Google Drive API

1. Go back to "APIs & Services" → "Library"
2. Search for "Google Drive API"
3. Click on it
4. Click "Enable"

---

## Step 3: Create Service Account

### 3.1: Create Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" at the top
3. Select "Service Account"
4. Fill in:
   - **Service account name**: `voice-agent-logger`
   - **Service account ID**: Auto-filled (leave as is)
   - **Description**: "Service account for voice agent call logging"
5. Click "Create and Continue"
6. Skip optional steps:
   - "Grant this service account access to project" → Click "Continue"
   - "Grant users access to this service account" → Click "Done"

### 3.2: Create Service Account Key

1. You'll see your service account in the list
2. Click on the service account email (looks like `voice-agent-logger@...`)
3. Go to the "Keys" tab
4. Click "Add Key" → "Create new key"
5. Select "JSON" format
6. Click "Create"
7. A JSON file will download automatically
8. **Keep this file safe** - it contains credentials

---

## Step 4: Prepare Credentials for Deployment

### 4.1: Open the Downloaded JSON File

Open the downloaded JSON file in a text editor. It should look like:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "voice-agent-logger@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}
```

### 4.2: Copy the Entire JSON Content

1. Select all the JSON content (Ctrl+A / Cmd+A)
2. Copy it (Ctrl+C / Cmd+C)
3. Keep it ready for the next step

**Important**: This JSON will go into your deployment environment variables as a single-line string.

---

## Step 5: Create Google Sheet

### Option A: Let the App Create It (Recommended)

The app will automatically create a sheet named "Voice Agent Call Logs" when it starts. Skip to Step 6.

### Option B: Create Manually (Optional)

1. Go to https://sheets.google.com
2. Create a new blank spreadsheet
3. Name it: "Voice Agent Call Logs"
4. Click "Share" button
5. Add the service account email (from the JSON file's `client_email` field)
   - Example: `voice-agent-logger@your-project.iam.gserviceaccount.com`
6. Give it "Editor" access
7. Click "Send"

---

## Step 6: Configure Deployment Environment Variables

Add these environment variables to your deployment platform (Render/Railway/Fly.io):

### 6.1: Add GOOGLE_SHEETS_CREDS_JSON

**Variable Name**: `GOOGLE_SHEETS_CREDS_JSON`

**Value**: The entire JSON content from Step 4.2 (paste as one line, with all quotes and special characters)

**Important**:
- This should be the complete JSON content as a string
- Don't remove quotes or escape characters
- The platform will handle it as a string

### 6.2: Add GOOGLE_SHEET_NAME (Optional)

**Variable Name**: `GOOGLE_SHEET_NAME`

**Value**: `Voice Agent Call Logs` (or your preferred sheet name)

**Note**: If you skip this, it defaults to "Voice Agent Call Logs"

### Example on Render:

1. Go to your service dashboard
2. Click "Environment" tab
3. Click "Add Environment Variable"
4. Key: `GOOGLE_SHEETS_CREDS_JSON`
5. Value: Paste the entire JSON
6. Click "Save Changes"
7. Repeat for `GOOGLE_SHEET_NAME` if you want a custom name

### Example on Railway:

1. Go to your project
2. Click on your service
3. Go to "Variables" tab
4. Click "New Variable"
5. Key: `GOOGLE_SHEETS_CREDS_JSON`
6. Value: Paste the entire JSON
7. Click "Add"

---

## Step 7: Deploy and Test

### 7.1: Deploy Your Changes

1. Make sure `requirements.txt` includes:
   ```
   gspread
   oauth2client
   ```
2. Push changes to your repository (if using Git deployment)
3. Or manually redeploy your service

### 7.2: Check Logs

After deployment:
1. Check your deployment logs
2. You should see: `"Google Sheets integration initialized successfully"`
3. If you see warnings about missing credentials, double-check Step 6

### 7.3: Make a Test Call

1. Call your Twilio number
2. Have a short conversation (30 seconds is fine)
3. End the call

### 7.4: Check Your Google Sheet

1. Go to https://sheets.google.com
2. Open "Voice Agent Call Logs" (or your custom name)
3. You should see a new row with:
   - Timestamp
   - Call SID
   - Phone numbers
   - Event type ("call_ended")
   - Duration
   - Number of exchanges
   - Summary of conversation

---

## Google Sheet Format

Your sheet will have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| Timestamp | When the call occurred | 2026-01-12 15:30:00 UTC |
| Call SID | Twilio call identifier | CA1234567890abcdef |
| From Number | Caller's phone number | +17862534432 |
| To Number | Your Twilio number | +15551234567 |
| Event Type | call_started or call_ended | call_ended |
| Duration (sec) | Call length in seconds | 125 |
| Exchanges | Number of conversation turns | 10 |
| Summary | AI-generated summary | Call duration: 10 exchanges. User spoke 5 times... |

---

## Troubleshooting

### Issue: "GOOGLE_SHEETS_CREDS_JSON not set" in logs

**Solution**:
- Verify environment variable is set in your platform
- Check variable name is exactly `GOOGLE_SHEETS_CREDS_JSON`
- Ensure the JSON was pasted completely

### Issue: "Permission denied" or "Insufficient permissions"

**Solutions**:
1. Verify Google Sheets API is enabled in Google Cloud Console
2. Verify Google Drive API is enabled
3. If you manually created the sheet, ensure service account email has Editor access
4. Try letting the app create the sheet automatically instead

### Issue: "Spreadsheet not found"

**Solutions**:
1. Let the app create it automatically (don't create manually)
2. If you created it manually, make sure:
   - Sheet name matches `GOOGLE_SHEET_NAME` environment variable
   - Service account email has access
   - Sheet is not in a shared drive (should be in "My Drive")

### Issue: No rows appear in sheet

**Solutions**:
1. Check deployment logs for errors
2. Verify calls are actually completing (check Twilio logs)
3. Make sure service account has "Editor" (not "Viewer") access
4. Try redeploying the service

### Issue: "Invalid JSON" errors

**Solution**:
- The JSON credentials might have been corrupted
- Re-download the JSON key from Google Cloud Console
- Copy-paste again carefully
- Ensure no extra spaces or newlines were added

---

## Security Best Practices

### Keep Credentials Safe

- **Never commit** the service account JSON to Git
- Store it only in environment variables on your deployment platform
- Don't share it publicly
- If compromised, delete the service account key and create a new one

### Limit Service Account Permissions

The service account should only have:
- Access to Google Sheets API
- Access to Google Drive API
- No other permissions needed

### Rotate Keys Periodically

Consider rotating service account keys every 6-12 months:
1. Create a new key
2. Update environment variable
3. Delete old key in Google Cloud Console

### Share Sheet Carefully

If sharing your call log sheet:
- Only share with trusted individuals
- Use "Viewer" access unless they need to edit
- Consider using a filtered view to hide sensitive info

---

## Advanced: Analyzing Your Call Data

### Filter and Sort

1. Click the filter icon in the header row
2. Filter by date range, phone number, or duration
3. Sort by any column to find patterns

### Create Charts

1. Select your data range
2. Insert → Chart
3. Create visualizations:
   - Calls per day (line chart)
   - Average call duration (bar chart)
   - Most frequent callers (pie chart)

### Export Data

1. File → Download
2. Choose format: Excel, CSV, PDF
3. Use for detailed analysis in other tools

### Set Up Formulas

Example formulas to add:

**Average Call Duration**:
```
=AVERAGE(F2:F)
```

**Total Calls Today**:
```
=COUNTIF(A:A, TODAY())
```

**Longest Call**:
```
=MAX(F2:F)
```

---

## Multiple Sheets for Different Purposes

You can create multiple sheets for different use cases:

1. **Call Logs**: Main log (auto-created)
2. **Summary**: Pivot tables and analysis
3. **Archive**: Move old calls here monthly

To add sheets:
1. Click the "+" at the bottom of your spreadsheet
2. Name it appropriately

---

## Backup and Data Retention

### Automatic Backups

Google Sheets automatically saves and versions your data. To view version history:
1. File → Version history → See version history
2. Restore any previous version if needed

### Manual Backups

For extra safety:
1. File → Download → Excel or CSV
2. Save to your computer monthly
3. Store in a secure location

### Data Retention

Consider your data retention policy:
- Keep all logs indefinitely (default)
- Archive logs older than 1 year to a separate sheet
- Delete logs older than 2 years if needed for privacy

---

## Cost

**Google Sheets API**:
- Free tier: 100 requests per 100 seconds
- Your app makes 1 request per call
- **Cost**: $0/month for typical usage

**Google Cloud Project**:
- No cost if only using Sheets API
- No credit card required for this usage level

**Total**: $0/month 🎉

---

## Next Steps

After setup:
1. ✅ Make multiple test calls to verify logging
2. ✅ Check sheet updates in real-time
3. ✅ Set up any custom formulas or charts
4. ✅ Consider sharing with team members (optional)
5. ✅ Set up periodic backups (optional but recommended)

---

## Support Resources

- **Google Sheets API Docs**: https://developers.google.com/sheets/api
- **Service Account Guide**: https://cloud.google.com/iam/docs/service-accounts
- **gspread Library Docs**: https://docs.gspread.org

---

**Congratulations!** Your call logs are now automatically saved to Google Sheets. You can access them anytime, anywhere! 📊✨
