"""
TALLI Server — Ollama-Compatible API
Routes to local Ollama models with task-aware features
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import argparse
import json

from .inference_engine import TALLIInference

app = FastAPI(title="TALLI Server")

# Global engine instance
engine: Optional[TALLIInference] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Dict = {}
    

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    options: Dict = {}


@app.on_event("startup")
async def startup():
    global engine
    print("🚀 Starting TALLI Server...")
    engine = TALLIInference(
        model_name=app.state.model_name,
        ollama_host=app.state.ollama_host,
    )
    print("✅ TALLI Server ready!")


@app.get("/api/tags")
async def list_models():
    """Ollama-compatible: list available models"""
    import requests
    try:
        resp = requests.get(f"{engine.ollama_host}/api/tags", timeout=10)
        return resp.json()
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Ollama-compatible: generate text"""
    import requests
    
    # Classify the task
    task_type = engine.classify_task(request.prompt)
    
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": request.stream,
        "options": request.options,
    }
    
    if request.stream:
        def stream_generator():
            resp = requests.post(
                f"{engine.ollama_host}/api/generate",
                json={**payload, "stream": True},
                stream=True,
                timeout=120
            )
            for line in resp.iter_lines():
                if line:
                    yield line.decode() + "\n"
        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        resp = requests.post(
            f"{engine.ollama_host}/api/generate",
            json=payload,
            timeout=120
        )
        result = resp.json()
        # Add TALLI metadata
        result["_talli"] = {
            "task_type": task_type,
            "routed_from": "talli",
        }
        return result


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Ollama-compatible: chat format"""
    import requests
    
    # Classify based on last user message
    last_user_msg = ""
    for msg in request.messages:
        if msg.role == "user":
            last_user_msg = msg.content
    
    task_type = engine.classify_task(last_user_msg) if last_user_msg else "chat"
    
    payload = {
        "model": request.model,
        "messages": [m.dict() for m in request.messages],
        "stream": request.stream,
        "options": request.options,
    }
    
    if request.stream:
        def stream_generator():
            resp = requests.post(
                f"{engine.ollama_host}/api/chat",
                json={**payload, "stream": True},
                stream=True,
                timeout=120
            )
            for line in resp.iter_lines():
                if line:
                    yield line.decode() + "\n"
        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        resp = requests.post(
            f"{engine.ollama_host}/api/chat",
            json=payload,
            timeout=120
        )
        result = resp.json()
        result["_talli"] = {
            "task_type": task_type,
            "routed_from": "talli",
        }
        return result


# TALLI-specific endpoints

@app.get("/api/talli/stats")
async def talli_stats():
    """TALLI-specific: get inference stats"""
    return engine.get_stats()


@app.get("/api/talli/classify")
async def talli_classify(query: str):
    """TALLI-specific: classify a query without generating"""
    task_type = engine.classify_task(query)
    return {
        "query": query,
        "task_type": task_type,
        "model": engine.get_model_for_task(task_type),
    }


@app.get("/api/talli/tasks")
async def talli_tasks():
    """TALLI-specific: list task types"""
    return {
        "tasks": [
            {"type": "code", "icon": "💻", "description": "Programming, debugging, code review"},
            {"type": "creative", "icon": "🎨", "description": "Writing, storytelling, brainstorming"},
            {"type": "reasoning", "icon": "🧠", "description": "Analysis, math, logic"},
            {"type": "chat", "icon": "💬", "description": "Casual conversation"},
            {"type": "multilingual", "icon": "🌍", "description": "Translation, multilingual"},
        ]
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "engine": "talli", "connected": True}


def main():
    parser = argparse.ArgumentParser(description="TALLI Server (Ollama)")
    parser.add_argument("--model", default="llama3.2", help="Default Ollama model")
    parser.add_argument("--port", type=int, default=11434, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--ollama", default="http://localhost:11434", help="Ollama host URL")
    args = parser.parse_args()
    
    app.state.model_name = args.model
    app.state.ollama_host = args.ollama
    
    print(f"🌐 TALLI Server starting on {args.host}:{args.port}")
    print(f"   Default model: {args.model}")
    print(f"   Ollama backend: {args.ollama}")
    print(f"   Ollama-compatible: YES")
    print(f"   Task-aware routing: YES")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
