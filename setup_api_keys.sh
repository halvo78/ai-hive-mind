#!/bin/bash
# Setup script for AI Hive Mind API keys

echo "============================================================"
echo "🔑 AI HIVE MIND API KEYS SETUP"
echo "============================================================"
echo ""
echo "This script will help you set up API keys for:"
echo "  1. OpenRouter (100+ FREE models)"
echo "  2. Groq (14,400 requests/day FREE)"
echo "  3. Hugging Face (30,000 chars/month FREE)"
echo ""
echo "All services are FREE - no credit card required!"
echo ""
echo "============================================================"
echo ""

# Function to set API key
set_api_key() {
    local service=$1
    local var_name=$2
    local url=$3
    
    echo "-----------------------------------------------------------"
    echo "Setting up $service"
    echo "-----------------------------------------------------------"
    echo ""
    echo "1. Visit: $url"
    echo "2. Sign up (FREE, no credit card)"
    echo "3. Get your API key"
    echo "4. Paste it below"
    echo ""
    read -p "Enter your $service API key (or press Enter to skip): " api_key
    
    if [ -n "$api_key" ]; then
        export $var_name="$api_key"
        echo "export $var_name='$api_key'" >> ~/.bashrc
        echo "✅ $service API key set!"
    else
        echo "⚠️ Skipped $service"
    fi
    echo ""
}

# OpenRouter
set_api_key "OpenRouter" "OPENROUTER_API_KEY" "https://openrouter.ai/keys"

# Groq
set_api_key "Groq" "GROQ_API_KEY" "https://console.groq.com/keys"

# Hugging Face
set_api_key "Hugging Face" "HF_TOKEN" "https://huggingface.co/settings/tokens"

echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "API keys have been saved to ~/.bashrc"
echo ""
echo "To use them now, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your terminal."
echo ""
echo "To test the AI Hive Mind, run:"
echo "  python3 /home/ubuntu/lyra-operational/ai/COMPLETE_FREE_AI_HIVE.py"
echo ""
echo "============================================================"

