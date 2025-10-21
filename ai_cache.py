
import hashlib
import time
from threading import Lock
from collections import OrderedDict

class SimpleCache:
    """Simple thread-safe LRU cache with TTL"""
    
    def __init__(self, max_size=1000, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = Lock()
    
    def _hash_key(self, prompt):
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt):
        """Get cached response if available and not expired"""
        with self.lock:
            key = self._hash_key(prompt)
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Expired
                    del self.cache[key]
            return None
    
    def set(self, prompt, response):
        """Cache response"""
        with self.lock:
            key = self._hash_key(prompt)
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (response, time.time())
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def stats(self):
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl
            }
