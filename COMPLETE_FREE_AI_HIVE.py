#!/usr/bin/env python3
"""
COMPLETE FREE AI HIVE MIND SYSTEM
Integrates ALL free resources: OpenRouter, Groq, Hugging Face, and more
Zero cost, maximum intelligence
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import asyncio
import aiohttp

class CompleteFreeAIHive:
    """
    Complete AI Hive Mind using ONLY free resources
    - OpenRouter (100+ free models)
    - Groq (14,400 requests/day)
    - Hugging Face (30,000 chars/month)
    - Together AI (free trial)
    - And more!
    """
    
    def __init__(self):
        # API Keys (get from environment or set here)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.hf_key = os.getenv("HF_TOKEN", "")
        self.together_key = os.getenv("TOGETHER_API_KEY", "")
        
        # Usage tracking
        self.groq_calls_today = 0
        self.groq_limit = 14400  # 14,400 requests/day
        self.hf_chars_used = 0
        self.hf_limit = 30000  # 30,000 chars/month
        self.last_reset = datetime.now()
        
        # OpenRouter FREE models (100+ available)
        self.openrouter_free_models = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
            "qwen/qwen-2.5-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "openchat/openchat-7b:free",
            "nousresearch/hermes-3-llama-3.1-8b:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free"
        ]
        
        # OpenRouter CHEAP models (for when you need more power)
        self.openrouter_cheap_models = [
            "deepseek/deepseek-chat",  # $0.14/$0.28 per 1M tokens
            "qwen/qwen-2.5-72b-instruct",  # $0.35/$0.40
            "meta-llama/llama-3.3-70b-instruct"  # $0.50/$0.75
        ]
        
        # Groq models (ultra-fast, free)
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma-2-9b-it"
        ]
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "groq_queries": 0,
            "openrouter_free_queries": 0,
            "openrouter_paid_queries": 0,
            "hf_queries": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0
        }
    
    def reset_daily_limits(self):
        """Reset daily limits if new day"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.groq_calls_today = 0
            self.last_reset = now
            print(f"✅ Daily limits reset at {now}")
    
    def query_groq(self, prompt: str, model: str = "llama-3.1-8b-instant", 
                   max_tokens: int = 1000) -> Tuple[Optional[str], str, float]:
        """
        Query Groq (FREE, ultra-fast)
        14,400 requests/day limit
        """
        self.reset_daily_limits()
        
        if not self.groq_key:
            return None, "groq_no_key", 0.0
        
        if self.groq_calls_today >= self.groq_limit:
            return None, "groq_limit_reached", 0.0
        
        try:
            start_time = time.time()
            
            # Use Groq API
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.groq_calls_today += 1
                self.stats["groq_queries"] += 1
                self.stats["total_queries"] += 1
                
                print(f"✅ Groq ({model}): {elapsed:.2f}s, {self.groq_calls_today}/{self.groq_limit} used")
                return content, f"groq_{model}", 0.0
            else:
                print(f"❌ Groq error: {response.status_code}")
                return None, "groq_error", 0.0
                
        except Exception as e:
            print(f"❌ Groq exception: {e}")
            return None, "groq_exception", 0.0
    
    def query_openrouter_free(self, prompt: str, model: Optional[str] = None,
                             max_tokens: int = 1000) -> Tuple[Optional[str], str, float]:
        """
        Query OpenRouter FREE models (100+ available)
        Unlimited requests with rate limits per model
        """
        if not self.openrouter_key:
            return None, "openrouter_no_key", 0.0
        
        # Try specified model or rotate through free models
        models_to_try = [model] if model else self.openrouter_free_models
        
        for model_name in models_to_try:
            try:
                start_time = time.time()
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/lyra-trading",
                        "X-Title": "Lyra Trading System"
                    },
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    self.stats["openrouter_free_queries"] += 1
                    self.stats["total_queries"] += 1
                    
                    print(f"✅ OpenRouter FREE ({model_name.split('/')[-1]}): {elapsed:.2f}s")
                    return content, f"openrouter_free_{model_name.split('/')[-1]}", 0.0
                else:
                    print(f"⚠️ OpenRouter {model_name}: {response.status_code}, trying next...")
                    continue
                    
            except Exception as e:
                print(f"⚠️ OpenRouter {model_name} exception: {e}, trying next...")
                continue
        
        return None, "openrouter_all_failed", 0.0
    
    def query_openrouter_cheap(self, prompt: str, model: Optional[str] = None,
                              max_tokens: int = 1000) -> Tuple[Optional[str], str, float]:
        """
        Query OpenRouter CHEAP models (when free isn't enough)
        DeepSeek V3: $0.14/$0.28 per 1M tokens (very cheap!)
        """
        if not self.openrouter_key:
            return None, "openrouter_no_key", 0.0
        
        model_name = model or self.openrouter_cheap_models[0]
        
        try:
            start_time = time.time()
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/lyra-trading",
                    "X-Title": "Lyra Trading System"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Calculate cost (approximate)
                input_tokens = len(prompt) / 4  # rough estimate
                output_tokens = len(content) / 4
                cost = (input_tokens * 0.14 + output_tokens * 0.28) / 1000000
                
                self.stats["openrouter_paid_queries"] += 1
                self.stats["total_queries"] += 1
                self.stats["total_cost"] += cost
                
                print(f"✅ OpenRouter CHEAP ({model_name.split('/')[-1]}): {elapsed:.2f}s, ${cost:.4f}")
                return content, f"openrouter_cheap_{model_name.split('/')[-1]}", cost
            else:
                print(f"❌ OpenRouter cheap error: {response.status_code}")
                return None, "openrouter_cheap_error", 0.0
                
        except Exception as e:
            print(f"❌ OpenRouter cheap exception: {e}")
            return None, "openrouter_cheap_exception", 0.0
    
    def query_huggingface(self, prompt: str, model: str = "meta-llama/Llama-3.2-3B-Instruct",
                         max_tokens: int = 500) -> Tuple[Optional[str], str, float]:
        """
        Query Hugging Face Inference API (FREE)
        30,000 characters/month limit
        """
        if not self.hf_key:
            return None, "hf_no_key", 0.0
        
        if self.hf_chars_used >= self.hf_limit:
            return None, "hf_limit_reached", 0.0
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {self.hf_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get('generated_text', '')
                    self.hf_chars_used += len(content)
                    self.stats["hf_queries"] += 1
                    self.stats["total_queries"] += 1
                    
                    print(f"✅ HuggingFace ({model.split('/')[-1]}): {elapsed:.2f}s, {self.hf_chars_used}/{self.hf_limit} chars")
                    return content, f"hf_{model.split('/')[-1]}", 0.0
                    
        except Exception as e:
            print(f"❌ HuggingFace exception: {e}")
        
        return None, "hf_error", 0.0
    
    def smart_query(self, prompt: str, importance: str = "low", 
                   max_tokens: int = 1000) -> Tuple[Optional[str], str, float]:
        """
        Smart routing based on importance
        - low: Groq or OpenRouter free
        - medium: OpenRouter free (multiple models)
        - high: OpenRouter cheap
        - critical: OpenRouter premium (not implemented yet)
        """
        self.reset_daily_limits()
        
        if importance == "low":
            # Try Groq first (fastest, free)
            result, source, cost = self.query_groq(prompt, max_tokens=max_tokens)
            if result:
                return result, source, cost
            
            # Fallback to OpenRouter free
            result, source, cost = self.query_openrouter_free(prompt, max_tokens=max_tokens)
            return result, source, cost
        
        elif importance == "medium":
            # Try OpenRouter free with multiple models
            result, source, cost = self.query_openrouter_free(prompt, max_tokens=max_tokens)
            if result:
                return result, source, cost
            
            # Fallback to Groq
            result, source, cost = self.query_groq(prompt, max_tokens=max_tokens)
            return result, source, cost
        
        elif importance == "high":
            # Use cheap models first
            result, source, cost = self.query_openrouter_cheap(prompt, max_tokens=max_tokens)
            if result:
                return result, source, cost
            
            # Fallback to free
            result, source, cost = self.query_openrouter_free(prompt, max_tokens=max_tokens)
            return result, source, cost
        
        else:
            # Default to free
            return self.query_openrouter_free(prompt, max_tokens=max_tokens)
    
    def multi_model_consensus(self, prompt: str, num_models: int = 3,
                             importance: str = "medium") -> Tuple[List[str], str, float]:
        """
        Get consensus from multiple FREE models
        Perfect for critical decisions
        """
        responses = []
        sources = []
        total_cost = 0.0
        
        print(f"\n🧠 Getting consensus from {num_models} models...")
        
        # Try Groq first (fastest)
        if self.groq_calls_today < self.groq_limit:
            result, source, cost = self.query_groq(prompt)
            if result:
                responses.append(result)
                sources.append(source)
                total_cost += cost
        
        # Get remaining from OpenRouter free models
        for i in range(num_models - len(responses)):
            if i < len(self.openrouter_free_models):
                model = self.openrouter_free_models[i]
                result, source, cost = self.query_openrouter_free(prompt, model=model)
                if result:
                    responses.append(result)
                    sources.append(source)
                    total_cost += cost
        
        if responses:
            print(f"✅ Got {len(responses)} responses, total cost: ${total_cost:.4f}")
            return responses, f"consensus_{len(responses)}_models", total_cost
        
        return [], "consensus_failed", 0.0
    
    def analyze_market(self, symbol: str, timeframe: str = "1h") -> Dict:
        """
        Analyze market using AI hive mind
        """
        prompt = f"""Analyze {symbol} market for {timeframe} timeframe.
        
Provide:
1. Trend direction (bullish/bearish/neutral)
2. Key support/resistance levels
3. Trading recommendation (buy/sell/hold)
4. Confidence level (0-100%)
5. Risk assessment

Be concise and specific."""
        
        print(f"\n📊 Analyzing {symbol} ({timeframe})...")
        
        # Get consensus from multiple models
        responses, source, cost = self.multi_model_consensus(prompt, num_models=3)
        
        if not responses:
            # Fallback to single query
            result, source, cost = self.smart_query(prompt, importance="medium")
            responses = [result] if result else []
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "responses": responses,
            "source": source,
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            **self.stats,
            "groq_remaining_today": self.groq_limit - self.groq_calls_today,
            "hf_chars_remaining": self.hf_limit - self.hf_chars_used,
            "avg_cost_per_query": self.stats["total_cost"] / max(self.stats["total_queries"], 1)
        }
    
    def print_stats(self):
        """Print usage statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 AI HIVE MIND STATISTICS")
        print("="*60)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"  - Groq: {stats['groq_queries']} (FREE)")
        print(f"  - OpenRouter Free: {stats['openrouter_free_queries']} (FREE)")
        print(f"  - OpenRouter Paid: {stats['openrouter_paid_queries']}")
        print(f"  - HuggingFace: {stats['hf_queries']} (FREE)")
        print(f"\nRemaining Today:")
        print(f"  - Groq: {stats['groq_remaining_today']}/{self.groq_limit}")
        print(f"  - HuggingFace: {stats['hf_chars_remaining']}/{self.hf_limit} chars")
        print(f"\nTotal Cost: ${stats['total_cost']:.4f}")
        print(f"Avg Cost/Query: ${stats['avg_cost_per_query']:.6f}")
        print("="*60 + "\n")


def main():
    """Demo and testing"""
    print("="*60)
    print("🚀 COMPLETE FREE AI HIVE MIND SYSTEM")
    print("="*60)
    print("\nIntegrating:")
    print("  ✅ OpenRouter (100+ FREE models)")
    print("  ✅ Groq (14,400 requests/day FREE)")
    print("  ✅ Hugging Face (30,000 chars/month FREE)")
    print("  ✅ Smart routing & consensus")
    print("\n" + "="*60 + "\n")
    
    # Initialize hive mind
    hive = CompleteFreeAIHive()
    
    # Check API keys
    print("🔑 Checking API keys...")
    if hive.openrouter_key:
        print("  ✅ OpenRouter API key found")
    else:
        print("  ⚠️ OpenRouter API key not found (set OPENROUTER_API_KEY)")
    
    if hive.groq_key:
        print("  ✅ Groq API key found")
    else:
        print("  ⚠️ Groq API key not found (set GROQ_API_KEY)")
    
    if hive.hf_key:
        print("  ✅ Hugging Face API key found")
    else:
        print("  ⚠️ Hugging Face API key not found (set HF_TOKEN)")
    
    print("\n" + "="*60 + "\n")
    
    # Test queries
    print("🧪 Testing AI Hive Mind...\n")
    
    # Test 1: Simple query (low importance)
    print("Test 1: Simple query (low importance)")
    result, source, cost = hive.smart_query(
        "What is Bitcoin?",
        importance="low",
        max_tokens=200
    )
    if result:
        print(f"Response: {result[:200]}...")
        print(f"Source: {source}, Cost: ${cost:.4f}\n")
    
    # Test 2: Market analysis (medium importance)
    print("\nTest 2: Market analysis (medium importance)")
    result, source, cost = hive.smart_query(
        "Analyze BTC/USD market trend for the next 24 hours",
        importance="medium",
        max_tokens=500
    )
    if result:
        print(f"Response: {result[:300]}...")
        print(f"Source: {source}, Cost: ${cost:.4f}\n")
    
    # Test 3: Multi-model consensus
    print("\nTest 3: Multi-model consensus (3 models)")
    responses, source, cost = hive.multi_model_consensus(
        "Should I buy BTC now? Give a yes/no answer with brief reasoning.",
        num_models=3
    )
    if responses:
        for i, resp in enumerate(responses, 1):
            print(f"\nModel {i}: {resp[:200]}...")
        print(f"\nSource: {source}, Total Cost: ${cost:.4f}\n")
    
    # Test 4: Full market analysis
    print("\nTest 4: Complete market analysis")
    analysis = hive.analyze_market("BTC-USD", "1h")
    print(f"Symbol: {analysis['symbol']}")
    print(f"Timeframe: {analysis['timeframe']}")
    print(f"Responses: {len(analysis['responses'])}")
    print(f"Cost: ${analysis['cost']:.4f}\n")
    
    # Print statistics
    hive.print_stats()
    
    print("✅ All tests complete!")
    print("\n" + "="*60)
    print("💡 USAGE TIPS:")
    print("="*60)
    print("1. Set API keys in environment variables:")
    print("   export OPENROUTER_API_KEY='your_key'")
    print("   export GROQ_API_KEY='your_key'")
    print("   export HF_TOKEN='your_key'")
    print("\n2. Use smart_query() for automatic routing:")
    print("   result, source, cost = hive.smart_query(prompt, importance='low')")
    print("\n3. Use multi_model_consensus() for critical decisions:")
    print("   responses, source, cost = hive.multi_model_consensus(prompt, num_models=3)")
    print("\n4. Check stats anytime:")
    print("   hive.print_stats()")
    print("\n5. Most queries cost $0 (using free models)!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

