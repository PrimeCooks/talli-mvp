"""
TALLI Server — Ollama-compatible API
Allows TALLI to be used as a drop-in Ollama replacement
"""
import os
import json
import time
import threading
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .inference_engine import TALLIInference


app = FastAPI(title="TALLI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[TALLIInference] = None
model_name = os.environ.get("TALLI_MODEL", "llama3-8b")


@app.on_event("startup")
async def startup():
    global engine
    print("🚀 Starting TALLI Server...")
    engine = TALLIInference(model_name=model_name, use_gpu=True)


@app.get("/")
async def root():
    return {"name": "TALLI", "version": "0.1.0", "status": "running"}


@app.get("/api/tags")
async def list_models():
    """Ollama-compatible: list available models"""
    return {
        "models": [
            {
                "name": f"talli-{model_name}",
                "model": f"talli-{model_name}",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": 5000000000,
                "digest": "sha256:talli-mvp",
                "details": {
                    "family": "talli",
                    "families": ["talli"],
                    "parameter_size": "8B",
                    "quantization_level": "FP16"
                }
            }
        ]
    }


@app.get("/api/show")
async def show_model(request: Request):
    """Ollama-compatible: show model info"""
    body = await request.json()
    name = body.get("name", model_name)
    return {
        "license": "Apache 2.0",
        "modelfile": f"FROM {name}",
        "parameters": "temperature 0.7",
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "family": "talli",
            "parameter_size": "8B"
        }
    }


@app.post("/api/generate")
async def generate(request: Request):
    """Ollama-compatible: text generation"""
    body = await request.json()
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    max_tokens = body.get("options", {}).get("num_predict", 256)
    temperature = body.get("options", {}).get("temperature", 0.7)
    
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)
    
    if stream:
        return StreamingResponse(
            stream_generate(prompt, max_tokens, temperature),
            media_type="application/x-ndjson"
        )
    
    # Non-streaming response
    response = engine.generate(
        query=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    
    task_type = engine.classify_task(prompt)
    
    return {
        "model": f"talli-{model_name}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
        "done": True,
        "context": [],
        "total_duration": 0,
        "eval_count": len(response.split()),
        "task_type": task_type,  # TALLI-specific
        "segments": engine.get_segment_layers(task_type),  # TALLI-specific
    }


async def stream_generate(prompt: str, max_tokens: int, temperature: float):
    """Stream generation response"""
    response = engine.generate(
        query=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    
    # Stream word by word
    words = response.split()
    for i, word in enumerate(words):
        chunk = {
            "model": f"talli-{model_name}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": word + " ",
            "done": False,
        }
        yield json.dumps(chunk) + "\n"
        time.sleep(0.01)
    
    # Final chunk
    yield json.dumps({
        "model": f"talli-{model_name}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": "",
        "done": True,
    }) + "\n"


@app.post("/api/chat")
async def chat(request: Request):
    """Ollama-compatible: chat format"""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)
    
    # Convert messages to prompt
    last_message = messages[-1].get("content", "")
    
    # Build context from previous messages
    context = ""
    for msg in messages[:-1]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        context += f"{role}: {content}\n"
    
    prompt = f"{context}user: {last_message}\nassistant:" if context else last_message
    
    response = engine.generate(query=prompt, max_new_tokens=256, temperature=0.7)
    task_type = engine.classify_task(last_message)
    
    return {
        "model": f"talli-{model_name}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {
            "role": "assistant",
            "content": response
        },
        "done": True,
        "task_type": task_type,
    }


# TALLI-specific endpoints

@app.get("/api/talli/stats")
async def talli_stats():
    """TALLI-specific: Get inference statistics"""
    if engine:
        return engine.get_stats()
    return {"error": "Engine not initialized"}


@app.get("/api/talli/segments/{task_type}")
async def talli_segments(task_type: str):
    """TALLI-specific: Get segments for a task type"""
    if engine:
        segments = engine.get_segment_layers(task_type)
        return {
            "task_type": task_type,
            "segments": segments,
            "attention_count": len(segments.get("attention_layers", [])),
            "ffn_count": len(segments.get("ffn_layers", [])),
        }
    return {"error": "Engine not initialized"}


@app.get("/api/talli/classify")
async def talli_classify(query: str):
    """TALLI-specific: Classify a query"""
    if engine:
        task_type = engine.classify_task(query)
        segments = engine.get_segment_layers(task_type)
        return {
            "query": query,
            "task_type": task_type,
            "segments_to_load": segments,
        }
    return {"error": "Engine not initialized"}


def run_server(model: str = "llama3-8b", host: str = "0.0.0.0", port: int = 11434):
    """Run the TALLI server"""
    import uvicorn
    global model_name
    model_name = model
    print(f"🌐 TALLI Server starting on {host}:{port}")
    print(f"   Model: {model}")
    print(f"   Ollama-compatible: YES")
    print(f"   Task-aware routing: YES")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
