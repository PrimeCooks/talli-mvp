"""
TALLI Inference Engine
Task-adaptive model loading and inference
"""
import torch
import os
from typing import Optional, Dict, List
from .task_router import TaskRouter
from .segment_index import SegmentIndex
from .memory_manager import MemoryManager


class TALLIInference:
    """Task-Adaptive LLM Inference Engine"""
    
    def __init__(self, model_name: str = "llama3-8b", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        print(f"🧠 TALLI Inference Engine")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   GPU: {torch.cuda.get_device_name(0) if self.use_gpu else 'None'}")
        
        # Initialize components
        self.task_router = TaskRouter()
        self.segment_index = SegmentIndex(model_name)
        self.memory_manager = MemoryManager(self.device)
        
        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Track loaded segments
        self.loaded_segments: Dict[str, List[int]] = {}
        self.current_task: Optional[str] = None
        
    def _load_model(self):
        """Load model based on configuration"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Map friendly names to HuggingFace model IDs
        MODEL_MAP = {
            "llama3-8b": "meta-llama/Meta-Llama-3-8B",
            "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
            "gemma-2b": "google/gemma-2b",
            "qwen2-7b": "Qwen/Qwen2-7B",
        }
        
        model_id = MODEL_MAP.get(self.model_name, self.model_name)
        
        print(f"   Loading model: {model_id}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        dtype = torch.float16 if self.use_gpu else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.use_gpu else None,
            low_cpu_mem_usage=True,
        )
        
        if not self.use_gpu:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print(f"   ✓ Model loaded")
        
        # Get total layers for segment mapping
        self.total_layers = self._count_layers()
        print(f"   Total layers: {self.total_layers}")
        
    def _count_layers(self) -> int:
        """Count transformer layers in the model"""
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return len(self.model.transformer.h)
            return 32  # Default for 8B models
        except:
            return 32
        
    def classify_task(self, query: str) -> str:
        """Classify the query into a task type"""
        return self.task_router.classify(query)
    
    def get_segment_layers(self, task_type: str) -> Dict[str, List[int]]:
        """Get which layers to activate for a task type"""
        return self.segment_index.get_segments(task_type, self.total_layers)
    
    def generate(self, query: str, max_new_tokens: int = 256, 
                 temperature: float = 0.7, stream: bool = False) -> str:
        """Generate response with task-adaptive loading"""
        
        # 1. Classify task
        task_type = self.classify_task(query)
        
        # 2. Get segments for this task
        segments = self.get_segment_layers(task_type)
        
        # 3. Log what we're doing
        active_layers = len(segments.get('attention_layers', []))
        total_attn = len(self.model.model.layers) if hasattr(self.model, 'model') else 32
        activation_pct = (active_layers / total_attn) * 100 if total_attn > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📝 Query: {query[:80]}...")
        print(f"🎯 Task: {task_type.upper()}")
        print(f"📊 Active layers: {active_layers}/{total_attn} ({activation_pct:.0f}%)")
        print(f"{'='*60}")
        
        # 4. Track segment loading
        self._update_loaded_segments(task_type, segments)
        
        # 5. Generate
        inputs = self.tokenizer(query, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 6. Log memory stats
        self._log_memory_stats(task_type)
        
        return response
    
    def _update_loaded_segments(self, task_type: str, segments: Dict[str, List[int]]):
        """Track which segments are loaded (for demonstration)"""
        if task_type != self.current_task:
            print(f"   🔄 Switching segments: {self.current_task} → {task_type}")
            self.loaded_segments = segments.copy()
            self.current_task = task_type
    
    def _log_memory_stats(self, task_type: str):
        """Log memory statistics"""
        if self.use_gpu:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        else:
            print(f"   💾 CPU mode (no GPU memory tracking)")
            
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            "model": self.model_name,
            "device": self.device,
            "current_task": self.current_task,
            "loaded_segments": self.loaded_segments,
            "total_layers": self.total_layers,
        }
