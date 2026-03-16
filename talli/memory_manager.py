"""
TALLI Memory Manager
Manages GPU/CPU memory and layer swapping
"""
import torch
from typing import Dict, List, Optional
from collections import OrderedDict
import time


class LRUCache:
    """LRU cache for segment eviction"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        
    def get(self, key: str) -> Optional[any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            
    def keys(self):
        return list(self.cache.keys())
    
    def __len__(self):
        return len(self.cache)


class MemoryManager:
    """Manages VRAM allocation and layer swapping"""
    
    def __init__(self, device: str = "cpu", max_vram_gb: float = 4.0):
        self.device = device
        self.max_vram_bytes = int(max_vram_gb * 1024**3)
        self.use_gpu = device == "cuda"
        
        # Track loaded layers
        self.loaded_layers: Dict[int, str] = {}  # layer_idx -> segment_name
        self.segment_cache = LRUCache(capacity=10)
        
        # Statistics
        self.stats = {
            "loads": 0,
            "evictions": 0,
            "swaps": 0,
            "total_bytes_loaded": 0,
        }
        
        print(f"💾 Memory Manager initialized")
        print(f"   Device: {device}")
        print(f"   Max VRAM: {max_vram_gb:.1f} GB")
        
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if not self.use_gpu:
            return float('inf')  # CPU has plenty
            
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / 1024**3
        return 0
    
    def get_used_memory(self) -> float:
        """Get used GPU memory in GB"""
        if not self.use_gpu:
            return 0
        return torch.cuda.memory_allocated() / 1024**3
    
    def estimate_layer_size(self, layer) -> float:
        """Estimate memory size of a layer in GB"""
        total_params = 0
        for param in layer.parameters():
            total_params += param.numel()
        bytes_per_param = 2 if param.dtype == torch.float16 else 4
        return (total_params * bytes_per_param) / 1024**3
    
    def can_load_segment(self, segment_layers: List[int], model) -> bool:
        """Check if we have room to load these layers"""
        if not self.use_gpu:
            return True  # CPU can handle anything
            
        estimated_size = 0
        for idx in segment_layers:
            if idx < len(model.model.layers):
                layer = model.model.layers[idx]
                estimated_size += self.estimate_layer_size(layer)
                
        available = self.get_available_memory() - self.get_used_memory()
        return estimated_size < (available * 0.8)  # Keep 20% buffer
    
    def load_layers_to_gpu(self, layer_indices: List[int], model, segment_name: str):
        """Load specific layers to GPU"""
        if not self.use_gpu:
            return
            
        for idx in layer_indices:
            if idx not in self.loaded_layers:
                if idx < len(model.model.layers):
                    layer = model.model.layers[idx]
                    layer.to("cuda")
                    self.loaded_layers[idx] = segment_name
                    self.stats["loads"] += 1
                    
        self.segment_cache.put(segment_name, layer_indices)
        
    def unload_layers_from_gpu(self, layer_indices: List[int], model):
        """Unload layers from GPU to CPU"""
        if not self.use_gpu:
            return
            
        for idx in layer_indices:
            if idx in self.loaded_layers:
                if idx < len(model.model.layers):
                    layer = model.model.layers[idx]
                    layer.to("cpu")
                    del self.loaded_layers[idx]
                    self.stats["evictions"] += 1
                    
    def ensure_segment_loaded(self, segment_name: str, layer_indices: List[int], 
                              model, evict_old: bool = True):
        """Ensure a segment is loaded, evicting old segments if needed"""
        
        # Check if already loaded
        if self.segment_cache.get(segment_name) is not None:
            return  # Already in cache
            
        # Check if we need to evict
        if not self.can_load_segment(layer_indices, model) and evict_old:
            # Find least recently used segment to evict
            for old_segment in list(self.segment_cache.keys()):
                old_layers = self.segment_cache.get(old_segment)
                if old_layers:
                    self.unload_layers_from_gpu(old_layers, model)
                    del self.segment_cache.cache[old_segment]
                    self.stats["swaps"] += 1
                    if self.can_load_segment(layer_indices, model):
                        break
                        
        # Load the new segment
        self.load_layers_to_gpu(layer_indices, model, segment_name)
        
    def get_stats(self) -> Dict:
        """Get memory manager statistics"""
        stats = self.stats.copy()
        if self.use_gpu:
            stats["gpu_memory_used_gb"] = self.get_used_memory()
            stats["gpu_memory_total_gb"] = self.get_available_memory()
            stats["loaded_layers_count"] = len(self.loaded_layers)
        return stats
        
    def cleanup(self, model):
        """Unload all layers from GPU"""
        if not self.use_gpu:
            return
            
        for idx in list(self.loaded_layers.keys()):
            if idx < len(model.model.layers):
                layer = model.model.layers[idx]
                layer.to("cpu")
                
        self.loaded_layers.clear()
        self.segment_cache.cache.clear()
        print("   🧹 Memory cleaned up")
