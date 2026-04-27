import re

with open("index.html", "r") as f:
    content = f.read()

# Add CSS for chat
chat_css = """
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 400px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            margin-top: 1rem;
        }

        .chat-messages {
            flex-grow: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .chat-message {
            max-width: 85%;
            padding: 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }

        .message-user {
            align-self: flex-end;
            background: var(--accent);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message-bot {
            align-self: flex-start;
            background: var(--bg-secondary);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 4px;
        }

        .chat-input-area {
            display: flex;
            padding: 1rem;
            background: rgba(30, 41, 59, 0.8);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            gap: 0.5rem;
        }

        .chat-input {
            flex-grow: 1;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 9999px;
            padding: 0.75rem 1.25rem;
            font-family: inherit;
            outline: none;
            transition: all 0.3s ease;
        }

        .chat-input:focus {
            border-color: var(--accent);
            background: rgba(0, 0, 0, 0.4);
        }

        .chat-submit {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 50%;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }

        .chat-submit:hover:not(:disabled) {
            background: var(--accent-hover);
            transform: scale(1.05);
        }

        .chat-submit:disabled {
            background: #475569;
            cursor: not-allowed;
            opacity: 0.7;
        }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            width: fit-content;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }
"""
content = content.replace("</style>", chat_css + "</style>")

# Replace the glass-container content
new_html = """
        <div class="glass-container">
            <h1>Resume AI Enhancer</h1>
            <p class="subtitle" id="main-subtitle">Upload your standard PDF resume and magic will be applied.</p>

            <div id="upload-section">
                <form id="upload-form">
                    <div class="upload-area" id="drop-zone">
                        <div class="upload-icon">📄</div>
                        <p>Drag & drop your PDF file here</p>
                        <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.5rem;">or click to browse</p>
                        <input type="file" id="file-input" accept=".pdf" aria-label="Resume PDF Upload">
                        <span class="file-name hidden" id="file-name-display"></span>
                    </div>

                    <div id="status-message" class="message-box hidden"></div>

                    <button type="submit" id="submit-btn" class="submit-btn" disabled>
                        <span class="btn-text">Enhance Resume</span>
                        <div class="loader"></div>
                    </button>
                </form>
            </div>

            <div id="chat-section" class="hidden">
                <div class="chat-container">
                    <div class="chat-messages" id="chat-messages">
                        <!-- Messages will appear here -->
                    </div>
                    <form class="chat-input-area" id="chat-form">
                        <input type="text" class="chat-input" id="chat-input" placeholder="Ask a question about the resume..." autocomplete="off" required>
                        <button type="submit" class="chat-submit" id="chat-submit-btn">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        </button>
                    </form>
                </div>
                <button id="restart-btn" class="submit-btn" style="margin-top: 1rem; background: var(--bg-secondary); border: 1px solid rgba(255,255,255,0.1);">Upload Another Resume</button>
            </div>
        </div>
"""

content = re.sub(r'<div class="glass-container">.*?</div>', new_html, content, flags=re.DOTALL)

# Add Javascript logic for chat
js_script = """
        const uploadSection = document.getElementById('upload-section');
        const chatSection = document.getElementById('chat-section');
        const chatMessages = document.getElementById('chat-messages');
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        const chatSubmitBtn = document.getElementById('chat-submit-btn');
        const restartBtn = document.getElementById('restart-btn');
        const mainSubtitle = document.getElementById('main-subtitle');

        restartBtn.addEventListener('click', () => {
            chatSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            fileInput.value = '';
            handleFileSelection(null);
            chatMessages.innerHTML = '';
            mainSubtitle.textContent = 'Upload your standard PDF resume and magic will be applied.';
        });

        function addChatMessage(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `chat-message message-${role}`;
            msgDiv.textContent = text;
            chatMessages.appendChild(msgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function showTypingIndicator() {
            const indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.id = 'typing-indicator';
            indicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
            chatMessages.appendChild(indicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function removeTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) {
                indicator.remove();
            }
        }

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;

            addChatMessage('user', message);
            chatInput.value = '';
            chatSubmitBtn.disabled = true;
            showTypingIndicator();

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('message', message);

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });

                removeTypingIndicator();

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Server error: ${response.status}`);
                }

                const data = await response.json();
                addChatMessage('bot', data.response);
                
            } catch (err) {
                removeTypingIndicator();
                addChatMessage('bot', `Error: ${err.message}`);
            } finally {
                chatSubmitBtn.disabled = false;
                chatInput.focus();
            }
        });
"""

# Inject new JS into existing script
content = content.replace("function hideMessage() {\n            statusMessage.className = 'message-box hidden';\n        }", "function hideMessage() {\n            statusMessage.className = 'message-box hidden';\n        }\n" + js_script)

# Also Modify the upload success logic to show chat
transition_logic = """
                // Trigger download
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                
                // Cleanup
                window.setTimeout(() => {
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }, 100);
                
                showMessage("Success! Your enhanced resume is downloading. Starting chat...", 'success');
                
                // Transition to chat
                setTimeout(() => {
                    uploadSection.classList.add('hidden');
                    chatSection.classList.remove('hidden');
                    mainSubtitle.textContent = `Chatting with ${file.name}`;
                    addChatMessage('bot', "Hello! I am ready to answer questions about this document. What would you like to know?");
                }, 1500);
"""

content = re.sub(r'// Trigger download.*?showMessage\("Success! Your enhanced resume is downloading\.", \'success\'\);', transition_logic, content, flags=re.DOTALL)

with open("index.html", "w") as f:
    f.write(content)
