"""
TALLI Web UI — Clean chat interface
"""
import os

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TALLI Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #fff; height: 100vh; display: flex; flex-direction: column; }
        
        /* Header */
        .header { padding: 16px 20px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 20px; font-weight: 600; }
        .header .status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #888; }
        .header .dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; }
        
        /* Model selector */
        .model-bar { padding: 12px 20px; background: #141414; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 12px; }
        .model-bar label { font-size: 13px; color: #888; }
        .model-bar select { background: #222; color: #fff; border: 1px solid #333; padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; }
        .task-badge { padding: 4px 10px; background: #333; border-radius: 12px; font-size: 12px; color: #aaa; }
        .task-badge.code { background: #1e3a5f; color: #60a5fa; }
        .task-badge.creative { background: #3b1f5e; color: #c084fc; }
        .task-badge.reasoning { background: #1f3d2b; color: #4ade80; }
        .task-badge.chat { background: #3d351f; color: #fbbf24; }
        
        /* Chat area */
        .chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message { max-width: 80%; padding: 14px 18px; border-radius: 16px; line-height: 1.5; font-size: 15px; }
        .message.user { align-self: flex-end; background: #3b82f6; color: white; border-bottom-right-radius: 4px; }
        .message.assistant { align-self: flex-start; background: #1a1a1a; border: 1px solid #333; border-bottom-left-radius: 4px; }
        .message pre { background: #111; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; font-size: 13px; }
        .message code { font-family: 'SF Mono', Monaco, monospace; }
        .message p { margin: 6px 0; }
        
        /* Input area */
        .input-area { padding: 16px 20px; background: #1a1a1a; border-top: 1px solid #333; }
        .input-row { display: flex; gap: 12px; align-items: flex-end; }
        .input-row textarea { flex: 1; background: #222; color: #fff; border: 1px solid #333; border-radius: 12px; padding: 12px 16px; font-size: 15px; resize: none; min-height: 48px; max-height: 150px; font-family: inherit; }
        .input-row textarea:focus { outline: none; border-color: #3b82f6; }
        .input-row button { background: #3b82f6; color: white; border: none; border-radius: 12px; padding: 12px 24px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        .input-row button:hover { background: #2563eb; }
        .input-row button:disabled { background: #333; cursor: not-allowed; }
        .hint { font-size: 12px; color: #555; margin-top: 8px; }
        
        /* Loading */
        .loading { display: flex; gap: 4px; padding: 8px 0; }
        .loading span { width: 8px; height: 8px; background: #3b82f6; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .loading span:nth-child(1) { animation-delay: -0.32s; }
        .loading span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 TALLI Chat</h1>
        <div class="status"><span class="dot"></span> Connected</div>
    </div>
    
    <div class="model-bar">
        <label>Model:</label>
        <select id="modelSelect"></select>
        <span id="taskBadge" class="task-badge chat">chat</span>
    </div>
    
    <div class="chat" id="chat"></div>
    
    <div class="input-area">
        <div class="input-row">
            <textarea id="input" placeholder="Ask me anything..." rows="1"></textarea>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
        <div class="hint">Shift+Enter for new line • Powered by TALLI + Ollama</div>
    </div>
    
    <script>
        let models = [];
        let currentModel = '';
        
        // Load models
        async function loadModels() {
            const resp = await fetch('/api/tags');
            const data = await resp.json();
            models = data.models || [];
            const select = document.getElementById('modelSelect');
            select.innerHTML = models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
            if (models.length) currentModel = models[0].name;
        }
        
        // Classify task
        async function classifyTask(query) {
            const resp = await fetch(`/api/talli/classify?query=${encodeURIComponent(query)}`);
            const data = await resp.json();
            const badge = document.getElementById('taskBadge');
            badge.textContent = `${data.task_type} → ${data.model.split(':')[0]}`;
            badge.className = `task-badge ${data.task_type}`;
        }
        
        // Add message to chat
        function addMessage(content, role) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = formatMessage(content);
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }
        
        // Simple markdown-ish formatting
        function formatMessage(text) {
            return text
                .replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\\n\\n/g, '</p><p>')
                .replace(/\\n/g, '<br>');
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            if (!text) return;
            
            const model = document.getElementById('modelSelect').value;
            input.value = '';
            input.style.height = '48px';
            
            addMessage(text, 'user');
            classifyTask(text);
            
            const btn = document.getElementById('sendBtn');
            btn.disabled = true;
            btn.textContent = '...';
            
            const loading = document.createElement('div');
            loading.className = 'message assistant';
            loading.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';
            document.getElementById('chat').appendChild(loading);
            
            try {
                const resp = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, prompt: text, stream: false })
                });
                const data = await resp.json();
                loading.remove();
                addMessage(data.response || 'No response', 'assistant');
            } catch (e) {
                loading.remove();
                addMessage(`Error: ${e.message}`, 'assistant');
            }
            
            btn.disabled = false;
            btn.textContent = 'Send';
        }
        
        // Input handling
        const input = document.getElementById('input');
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        input.addEventListener('input', () => {
            input.style.height = '48px';
            input.style.height = Math.min(input.scrollHeight, 150) + 'px';
        });
        
        // Init
        loadModels();
    </script>
</body>
</html>"""

def get_ui_html():
    return UI_HTML
