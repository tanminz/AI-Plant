# OpenAI API Integration Setup

## Overview

The Plant AI System now integrates with OpenAI's API to provide detailed, AI-powered treatment recommendations for plant diseases. This includes:

- **Treatment Steps**: Step-by-step treatment procedures
- **Medicine Recommendations**: Specific medicines with dosages and application methods
- **Care Instructions**: Detailed care guidelines
- **Recovery Timeline**: Expected recovery timeframes
- **Monitoring Guidelines**: What to watch for during treatment

## Setup Instructions

### 1. Install Required Packages

```bash
pip install openai python-dotenv
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy your API key (you won't be able to see it again!)

### 3. Configure API Key

#### Option A: Using .env file (Recommended)

1. Create a `.env` file in the `plant_ai_system` directory:
```bash
cd plant_ai_system
touch .env
```

2. Add your API key to the `.env` file:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

3. The `.env` file is already in `.gitignore`, so your key won't be committed to git.

#### Option B: Environment Variable

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 4. Verify Setup

When you start the Flask app, you should see:
```
OpenAI API initialized successfully
```

If you see a warning instead, check:
- API key is correctly set
- OpenAI package is installed
- Internet connection is available

## Usage

Once configured, the OpenAI integration works automatically:

1. Upload a plant image
2. The system detects the disease
3. If treatment is needed, OpenAI API is called automatically
4. Detailed treatment advice is displayed in the results

## Features

### AI-Powered Treatment Overview
Provides a comprehensive overview of the disease and treatment approach.

### Treatment Steps
Step-by-step instructions for treating the disease.

### Medicine Recommendations
- Medicine name and type
- Dosage information
- Application method
- Frequency of application
- Safety precautions

### Care Instructions
Detailed care guidelines during treatment.

### Recovery Timeline
Expected time for plant recovery.

### Monitoring Guidelines
What to watch for during the treatment process.

## Cost Considerations

- The system uses `gpt-4o-mini` model for cost efficiency
- Each disease analysis makes one API call
- Approximate cost: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- Typical response: ~500-1000 tokens (~$0.001 per analysis)

## Troubleshooting

### "OpenAI API not initialized"
- Check that your API key is set correctly
- Verify the `.env` file is in the correct location
- Restart the Flask application

### "Error calling OpenAI API"
- Check your internet connection
- Verify your API key is valid and has credits
- Check OpenAI API status: https://status.openai.com/

### No OpenAI advice shown
- OpenAI advice only appears when treatment is needed
- Healthy plants won't trigger OpenAI calls
- Check browser console for errors

## Security Notes

- **Never commit your API key to git**
- The `.env` file is already in `.gitignore`
- Keep your API key secret and secure
- Rotate your API key if it's exposed

## Model Configuration

By default, the system uses `gpt-4o-mini`. To change the model, modify the `get_openai_treatment_advice` function in `app.py`:

```python
response = openai_client.chat.completions.create(
    model="gpt-4",  # Change to gpt-4, gpt-3.5-turbo, etc.
    ...
)
```

## Support

For issues or questions:
1. Check OpenAI API documentation: https://platform.openai.com/docs
2. Review error messages in the Flask console
3. Check browser console for frontend errors

---

**Note**: The system gracefully handles cases where OpenAI API is not available. Basic treatment recommendations from the disease database will still be shown.

