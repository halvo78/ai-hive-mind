# 🚀 QUICK START GUIDE - AI HIVE MIND

## Get Started in 5 Minutes!

### Step 1: Get FREE API Keys (3 minutes)

#### OpenRouter (100+ FREE models)
1. Visit: https://openrouter.ai/keys
2. Sign up (FREE, no credit card)
3. Click "Create Key"
4. Copy your API key

#### Groq (14,400 requests/day FREE)
1. Visit: https://console.groq.com/keys
2. Sign up (FREE, no credit card)
3. Click "Create API Key"
4. Copy your API key

#### Hugging Face (Optional - 30,000 chars/month FREE)
1. Visit: https://huggingface.co/settings/tokens
2. Sign up (FREE)
3. Click "New token"
4. Copy your token

---

### Step 2: Set API Keys (1 minute)

**Option A: Interactive Setup**
```bash
cd /home/ubuntu/lyra-operational/ai
./setup_api_keys.sh
```

**Option B: Manual Setup**
```bash
export OPENROUTER_API_KEY='your_openrouter_key_here'
export GROQ_API_KEY='your_groq_key_here'
export HF_TOKEN='your_huggingface_token_here'

# Save to .bashrc for persistence
echo "export OPENROUTER_API_KEY='your_key'" >> ~/.bashrc
echo "export GROQ_API_KEY='your_key'" >> ~/.bashrc
echo "export HF_TOKEN='your_key'" >> ~/.bashrc
```

---

### Step 3: Test the System (1 minute)

```bash
cd /home/ubuntu/lyra-operational/ai
python3 COMPLETE_FREE_AI_HIVE.py
```

You should see:
- ✅ API keys detected
- 🧪 Running tests
- 📊 Statistics
- ✅ All tests complete!

---

## Usage Examples

### Basic Query
```python
from COMPLETE_FREE_AI_HIVE import CompleteFreeAIHive

hive = CompleteFreeAIHive()

# Simple query (FREE)
result, source, cost = hive.smart_query(
    "What is Bitcoin?",
    importance="low"
)
print(f"Answer: {result}")
print(f"Cost: ${cost}")  # Usually $0!
```

### Market Analysis
```python
# Analyze any market (FREE)
analysis = hive.analyze_market("BTC-USD", "1h")
print(f"Analysis: {analysis}")
```

### Multi-Model Consensus
```python
# Get consensus from 3 models (FREE)
responses, source, cost = hive.multi_model_consensus(
    "Should I buy BTC now?",
    num_models=3
)

for i, response in enumerate(responses, 1):
    print(f"Model {i}: {response}")

print(f"Total cost: ${cost}")  # Usually $0!
```

### Check Statistics
```python
# See your usage
hive.print_stats()
```

---

## What You Get

### FREE Resources:
- ✅ **OpenRouter:** 100+ models, unlimited requests*
- ✅ **Groq:** 14,400 requests/day, ultra-fast
- ✅ **Hugging Face:** 30,000 chars/month
- ✅ **Smart routing:** Automatic fallback
- ✅ **Multi-model consensus:** Multiple AI opinions
- ✅ **Cost tracking:** Know exactly what you spend

*With rate limits per model

### Cost:
- **95% of queries:** $0 (using free models)
- **5% of queries:** $0.0001-0.001 (using cheap models)
- **Average:** $0-5/month for 3,000 trades

---

## Importance Levels

### Low (FREE)
- Uses Groq or OpenRouter free models
- Perfect for: Quick checks, monitoring, simple questions
- Cost: $0

### Medium (FREE)
- Uses OpenRouter free models with fallback
- Perfect for: Analysis, research, decision support
- Cost: $0

### High (CHEAP)
- Uses OpenRouter cheap models (DeepSeek V3)
- Perfect for: Complex analysis, critical decisions
- Cost: $0.0001-0.001 per query

---

## Tips for Maximum FREE Usage

1. **Use 'low' importance for most queries** - It's free!
2. **Use multi_model_consensus() for critical decisions** - Still free!
3. **Groq is fastest** - Use it for real-time needs
4. **OpenRouter has 100+ free models** - Unlimited variety
5. **Check stats regularly** - Monitor your usage

---

## Troubleshooting

### "No API key found"
- Make sure you set the environment variables
- Run: `source ~/.bashrc` to reload
- Or restart your terminal

### "Rate limit reached"
- Groq: Wait until tomorrow (resets daily)
- OpenRouter: Try a different free model
- System automatically handles this!

### "All models failed"
- Check your internet connection
- Verify API keys are correct
- Try running the test script again

---

## Next Steps

1. ✅ Get API keys (3 min)
2. ✅ Run test script (1 min)
3. ✅ Integrate with Lyra trading system
4. ✅ Start trading with AI assistance!

---

## Support

- **Documentation:** See COMPLETE_FREE_GPU_AI_GUIDE.md
- **Code:** /home/ubuntu/lyra-operational/ai/COMPLETE_FREE_AI_HIVE.py
- **Issues:** Check the logs and error messages

---

**You're now ready to use the world's most powerful FREE AI hive mind!** 🚀

**Cost: $0-5/month instead of $5,000+/month!**

