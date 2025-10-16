# 🔧 API Key Configuration Guide

## Current Status
✅ **OpenRouter API Key**: Configured  
⚠️ **OpenAI API Key**: Placeholder set (needs real key for embeddings)

## How to Get Your OpenAI API Key

### Step 1: Visit OpenAI Platform
Go to: https://platform.openai.com/api-keys

### Step 2: Sign Up/Login
- Create account or login with existing account
- You may need to add billing information

### Step 3: Create API Key
1. Click "Create new secret key"
2. Give it a name (e.g., "Agentic Graph RAG")
3. Copy the key (starts with `sk-proj-` or `sk-`)

### Step 4: Update Your .env File
Replace `your_openai_key_here` with your actual key:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

## Alternative: Run Without Embeddings

If you don't want to use OpenAI, you can:

### Option 1: Disable Embeddings
In your `.env` file, set:
```
ENABLE_EMBEDDINGS=false
```

### Option 2: Use Only Basic Processing
The system will work with:
- ✅ Document processing
- ✅ LLM ontology generation (OpenRouter)
- ✅ Basic graph visualization
- ❌ Vector embeddings (requires OpenAI)
- ❌ Semantic similarity search

## Testing Your Configuration

Run this to check your setup:
```bash
python test_phase2.py
```

## Launching the GUI

Once configured, launch with:
```bash
python phase2_gui.py
```

The warning dialogs will disappear once you have valid API keys set!