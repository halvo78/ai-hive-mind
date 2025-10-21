
import time
from collections import defaultdict
from datetime import datetime
import json

class PerformanceMonitor:
    """Monitor AI Hive performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'queries_total': 0,
            'queries_success': 0,
            'queries_failed': 0,
            'total_cost': 0.0,
            'total_latency': 0.0,
            'by_source': defaultdict(lambda: {
                'count': 0,
                'cost': 0.0,
                'latency': 0.0
            }),
            'start_time': datetime.now().isoformat()
        }
    
    def record_query(self, source, cost, latency, success):
        """Record query metrics"""
        self.metrics['queries_total'] += 1
        
        if success:
            self.metrics['queries_success'] += 1
        else:
            self.metrics['queries_failed'] += 1
        
        self.metrics['total_cost'] += cost
        self.metrics['total_latency'] += latency
        
        self.metrics['by_source'][source]['count'] += 1
        self.metrics['by_source'][source]['cost'] += cost
        self.metrics['by_source'][source]['latency'] += latency
    
    def get_stats(self):
        """Get current statistics"""
        stats = dict(self.metrics)
        
        if stats['queries_total'] > 0:
            stats['avg_cost'] = stats['total_cost'] / stats['queries_total']
            stats['avg_latency'] = stats['total_latency'] / stats['queries_total']
            stats['success_rate'] = (stats['queries_success'] / stats['queries_total']) * 100
        else:
            stats['avg_cost'] = 0
            stats['avg_latency'] = 0
            stats['success_rate'] = 0
        
        return stats
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Queries: {stats['queries_total']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Avg Latency: {stats['avg_latency']:.2f}s")
        print(f"Avg Cost: ${stats['avg_cost']:.6f}")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print("\nBy Source:")
        for source, data in stats['by_source'].items():
            print(f"  {source}: {data['count']} queries, "
                  f"${data['cost']:.4f}, "
                  f"{data['latency']/data['count']:.2f}s avg")
        print("="*60 + "\n")
    
    def save_metrics(self, filename='metrics.json'):
        """Save metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.get_stats(), f, indent=2, default=str)
