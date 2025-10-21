
import asyncio
import aiohttp
from typing import List, Tuple

class AsyncAIHive:
    """Async version of AI Hive for parallel queries"""
    
    def __init__(self, api_keys):
        self.api_keys = api_keys
    
    async def query_async(self, session, prompt, model, api_key):
        """Query AI model asynchronously"""
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'], model, 0.0
        except Exception as e:
            return None, model, 0.0
        
        return None, model, 0.0
    
    async def multi_query_parallel(self, prompts: List[str]) -> List[Tuple]:
        """Query multiple prompts in parallel"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in prompts:
                task = self.query_async(
                    session,
                    prompt,
                    "meta-llama/llama-3.2-3b-instruct:free",
                    self.api_keys['openrouter']
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
    
    def query_parallel(self, prompts: List[str]) -> List[Tuple]:
        """Synchronous wrapper for parallel queries"""
        return asyncio.run(self.multi_query_parallel(prompts))
