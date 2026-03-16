"""
TALLI CLI — Interactive chat with task-adaptive inference
"""
import argparse
import sys
import time


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  TALLI — Task-Adaptive LLM Inference                        ║
║  Only load the model segments you need                      ║
║  /quit to exit  /stats for memory info  /segments to see    ║
╚══════════════════════════════════════════════════════════════╝
"""

TASK_ICONS = {
    "code": "💻",
    "creative": "🎨",
    "reasoning": "🧠",
    "chat": "💬",
    "multilingual": "🌍",
}


def run_cli(model: str = "llama3-8b", use_gpu: bool = True):
    """Run interactive CLI"""
    print(BANNER)
    
    from .inference_engine import TALLIInference
    
    print(f"Loading {model}...")
    engine = TALLIInference(model_name=model, use_gpu=use_gpu)
    print(f"\nReady! Type your questions below.\n")
    
    while True:
        try:
            # Get input
            query = input("You: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ["/quit", "/exit", "/q"]:
                print("\n👋 Goodbye!")
                break
                
            if query.lower() == "/stats":
                stats = engine.get_stats()
                print(f"\n📊 TALLI Stats:")
                for k, v in stats.items():
                    if k != "loaded_segments":
                        print(f"   {k}: {v}")
                print()
                continue
                
            if query.lower() == "/segments":
                print(f"\n📦 Available segments:")
                for task in ["code", "creative", "reasoning", "chat", "multilingual"]:
                    seg = engine.get_segment_layers(task)
                    attn = len(seg.get("attention_layers", []))
                    ffn = len(seg.get("ffn_layers", []))
                    icon = TASK_ICONS.get(task, "❓")
                    print(f"   {icon} {task:12s} → {attn + ffn} layers ({attn} attn, {ffn} ffn)")
                print()
                continue
            
            if query.lower() == "/model":
                print(f"\n🤖 Current model: {model}")
                print(f"   Device: {engine.device}")
                print(f"   Total layers: {engine.total_layers}")
                print()
                continue
            
            if query.lower() == "/help":
                print(f"\n📖 Commands:")
                print(f"   /quit, /exit, /q  — Exit the chat")
                print(f"   /stats            — Show memory/GPU stats")
                print(f"   /segments         — Show layer segments")
                print(f"   /model            — Show model info")
                print(f"   /help             — Show this help")
                print()
                continue
                
            # Generate response
            task_type = engine.classify_task(query)
            icon = TASK_ICONS.get(task_type, "❓")
            
            print(f"\n{icon} Task: {task_type}")
            print("TALLI: ", end="", flush=True)
            
            start = time.time()
            response = engine.generate(query, max_new_tokens=256)
            elapsed = time.time() - start
            
            print(response)
            print(f"\n⏱️  {elapsed:.2f}s | Task: {task_type}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="TALLI CLI — Task-Adaptive LLM Inference")
    parser.add_argument("--model", default="llama3-8b", help="Model to use (default: llama3-8b)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (use CPU only)")
    args = parser.parse_args()
    
    run_cli(model=args.model, use_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
