"""
TALLI Inference Engine — Ollama Edition
Uses local Ollama models instead of downloading from HuggingFace
"""
import requests
import json
import os
from typing import Optional, Dict, List, Generator
from .task_router import TaskRouter


# Map task types to preferred models
TASK_MODEL_MAP = {
    "code": None,        # Use default or user-specified
    "creative": None,
    "reasoning": None,
    "chat": None,
    "multilingual": None,
}

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class TALLIInference:
    """Task-Adaptive LLM using Ollama models"""
    
    def __init__(self, model_name: str = "llama3.2", ollama_host: str = None, use_gpu: bool = True):
        self.model_name = model_name
        self.ollama_host = ollama_host or OLLAMA_BASE
        self.device = "gpu"  # Ollama handles GPU/CPU automatically
        
        print(f"🧠 TALLI Inference Engine (Ollama)")
        print(f"   Model: {model_name}")
        print(f"   Ollama: {self.ollama_host}")
        
        # Initialize components
        self.task_router = TaskRouter()
        
        # Verify Ollama connection
        self._check_ollama()
        
        # Track current task
        self.current_task: Optional[str] = None
        self.total_layers = 32  # Default, not used for routing
        
    def _check_ollama(self):
        """Verify Ollama is running and has models"""
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                print(f"   ✓ Ollama connected ({len(models)} models available)")
                for m in models[:3]:
                    print(f"     • {m['name']}")
                if len(models) > 3:
                    print(f"     ... and {len(models)-3} more")
            else:
                print(f"   ⚠️ Ollama returned status {resp.status_code}")
        except Exception as e:
            print(f"   ❌ Cannot connect to Ollama: {e}")
            print(f"   Make sure Ollama is running at {self.ollama_host}")
            raise
            
    def _list_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except:
            pass
        return []
        
    def classify_task(self, query: str) -> str:
        """Classify the query into a task type"""
        return self.task_router.classify(query)
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get the best model for this task type"""
        # If user set a specific model for this task, use it
        if TASK_MODEL_MAP.get(task_type):
            return TASK_MODEL_MAP[task_type]
        return self.model_name
    
    def generate(self, query: str, max_new_tokens: int = 256, 
                 temperature: float = 0.7, stream: bool = False) -> str:
        """Generate response using Ollama"""
        
        # 1. Classify task
        task_type = self.classify_task(query)
        
        # 2. Get model for this task
        model = self.get_model_for_task(task_type)
        
        # 3. Log what we're doing
        print(f"\n{'='*60}")
        print(f"📝 Query: {query[:80]}...")
        print(f"🎯 Task: {task_type.upper()}")
        print(f"🤖 Model: {model}")
        print(f"{'='*60}")
        
        # 4. Track task switch
        if task_type != self.current_task:
            if self.current_task:
                print(f"   🔄 Task switch: {self.current_task} → {task_type}")
            self.current_task = task_type
        
        # 5. Generate via Ollama
        payload = {
            "model": model,
            "prompt": query,
            "stream": stream,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            }
        }
        
        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_sync(payload)
    
    def _generate_sync(self, payload: dict) -> str:
        """Synchronous generation"""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("response", "")
            else:
                return f"Error: Ollama returned {resp.status_code}"
        except Exception as e:
            return f"Error: {e}"
    
    def _generate_stream(self, payload: dict) -> Generator[str, None, None):
        """Streaming generation"""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={**payload, "stream": True},
                stream=True,
                timeout=120
            )
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
        except Exception as e:
            yield f"Error: {e}"
    
    def chat(self, messages: List[Dict], model: str = None, stream: bool = False) -> str:
        """Chat format generation (Ollama /api/chat)"""
        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": stream,
        }
        
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
            else:
                return f"Error: {resp.status_code}"
        except Exception as e:
            return f"Error: {e}"
            
    def get_stats(self) -> Dict:
        """Get current statistics"""
        models = self._list_models()
        return {
            "model": self.model_name,
            "device": self.device,
            "current_task": self.current_task,
            "ollama_host": self.ollama_host,
            "available_models": models,
        }
