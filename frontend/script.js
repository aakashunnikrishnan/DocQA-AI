/**
 * DocQA AI - Frontend JavaScript
 * Handles chat, document upload, and API interactions
 * FIXED: UI bugs, error handling, message rendering, and connection management
 */

// ========================================
// State
// ========================================
const state = {
    documents: [],
    queryCount: 0,
    isProcessing: false,
    apiBase: 'http://localhost:8000/api/v1',
    ws: null,
    isConnected: false,
    theme: 'light',
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    reconnectDelay: 2000,
    messageQueue: [],
    isStreaming: false,
    currentStreamMessage: null
};

// ========================================
// DOM References
// ========================================
const DOM = {
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    documentList: document.getElementById('documentList'),
    statusBadge: document.getElementById('statusBadge'),
    uploadModal: document.getElementById('uploadModal'),
    fileInput: document.getElementById('fileInput'),
    uploadArea: document.getElementById('uploadArea'),
    uploadProgress: document.getElementById('uploadProgress'),
    uploadProgressFill: document.getElementById('uploadProgressFill'),
    uploadStatus: document.getElementById('uploadStatus'),
    uploadedFiles: document.getElementById('uploadedFiles'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    statDocuments: document.getElementById('statDocuments'),
    statChunks: document.getElementById('statChunks'),
    statQueries: document.getElementById('statQueries'),
    docCountHeader: document.getElementById('docCountHeader'),
    charCount: document.getElementById('charCount'),
    includeSources: document.getElementById('includeSources'),
    topKSlider: document.getElementById('topKSlider'),
    temperatureSlider: document.getElementById('temperatureSlider'),
    uploadBtn: document.getElementById('uploadBtn')
};

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    init();
});

function init() {
    // Load theme preference
    const savedTheme = localStorage.getItem('docqa-theme') || 'light';
    setTheme(savedTheme);

    // Check API health
    checkHealth();

    // Load documents
    loadDocuments();

    // Setup event listeners
    setupEventListeners();

    // Setup WebSocket
    setupWebSocket();

    // Auto-resize textarea
    DOM.chatInput.addEventListener('input', autoResizeTextarea);
    DOM.chatInput.addEventListener('input', updateCharCount);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Initial UI state
    updateUIState();

    console.log('📄 DocQA AI frontend initialized');
}

// ========================================
// Event Listeners
// ========================================
function setupEventListeners() {
    // File upload
    DOM.uploadArea.addEventListener('dragover', handleDragOver);
    DOM.uploadArea.addEventListener('dragleave', handleDragLeave);
    DOM.uploadArea.addEventListener('drop', handleDrop);
    DOM.uploadArea.addEventListener('click', () => DOM.fileInput.click());
    DOM.fileInput.addEventListener('change', handleFileSelect);

    // Modal close
    DOM.uploadModal.addEventListener('click', (e) => {
        if (e.target === DOM.uploadModal) {
            closeUploadModal();
        }
    });

    // Upload button
    DOM.uploadBtn.addEventListener('click', uploadFiles);

    // Send button
    DOM.sendBtn.addEventListener('click', sendMessage);

    // Chat input enter key
    DOM.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Settings changes
    DOM.includeSources.addEventListener('change', updateUIState);
    DOM.topKSlider.addEventListener('input', updateUIState);
    DOM.temperatureSlider.addEventListener('input', updateUIState);
}

// ========================================
// Keyboard Shortcuts
// ========================================
function handleKeyboardShortcuts(e) {
    // Ctrl+K or Cmd+K to focus chat input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        DOM.chatInput.focus();
    }

    // Escape to close modal
    if (e.key === 'Escape') {
        closeUploadModal();
    }

    // Ctrl+Enter to send
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendMessage();
    }
}

// ========================================
// API Functions
// ========================================
async function apiRequest(endpoint, method = 'GET', data = null) {
    const url = `${state.apiBase}${endpoint}`;
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(url, options);

        // Handle non-JSON responses
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return { success: true };
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || result.message || `HTTP ${response.status}`);
        }

        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function checkHealth() {
    try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
            updateStatus('connected');
            console.log('✅ API is healthy');
            return true;
        } else {
            updateStatus('disconnected');
            console.warn('⚠️ API is not responding');
            return false;
        }
    } catch (error) {
        updateStatus('disconnected');
        console.warn('⚠️ Cannot connect to API:', error.message);
        return false;
    }
}

async function loadDocuments() {
    try {
        const data = await apiRequest('/documents');
        state.documents = data.documents || [];
        renderDocumentList();
        updateStats();
    } catch (error) {
        console.error('Failed to load documents:', error);
        DOM.documentList.innerHTML = `
            <div class="document-item" style="color:var(--danger);">
                <span>❌</span>
                <span>Error loading documents: ${error.message}</span>
            </div>
        `;
    }
}

async function uploadFiles() {
    const files = DOM.fileInput.files;
    if (files.length === 0) {
        showNotification('Please select files to upload', 'warning');
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    // Add chunking parameters
    formData.append('chunk_size', '800');
    formData.append('chunk_overlap', '150');
    formData.append('chunking_strategy', 'adaptive');

    DOM.uploadProgress.style.display = 'block';
    DOM.uploadProgressFill.style.width = '0%';
    DOM.uploadStatus.textContent = 'Uploading...';
    DOM.uploadBtn.disabled = true;
    DOM.uploadBtn.textContent = 'Uploading...';

    try {
        const response = await fetch(`${state.apiBase}/documents/ingest`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            let errorMessage = 'Upload failed';
            try {
                const error = await response.json();
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();

        DOM.uploadProgressFill.style.width = '100%';
        DOM.uploadStatus.textContent = '✅ Upload complete!';

        // Show uploaded files
        renderUploadedFiles(files, result);

        // Reload documents
        await loadDocuments();

        // Show success notification
        showNotification(
            `✅ Successfully ingested ${result.total_documents} documents (${result.total_chunks} chunks)`,
            'success'
        );

        // Close modal after delay
        setTimeout(() => {
            closeUploadModal();
            DOM.uploadProgress.style.display = 'none';
            DOM.uploadBtn.disabled = false;
            DOM.uploadBtn.textContent = 'Upload';
        }, 1500);

    } catch (error) {
        DOM.uploadStatus.textContent = `❌ Error: ${error.message}`;
        DOM.uploadBtn.disabled = false;
        DOM.uploadBtn.textContent = 'Upload';
        showNotification(`Upload failed: ${error.message}`, 'error');
        console.error('Upload error:', error);
    }
}

// ========================================
// Message Functions
// ========================================
async function sendMessage() {
    const question = DOM.chatInput.value.trim();
    if (!question || state.isProcessing) return;

    // Clear input
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    updateCharCount();

    // Add user message
    addMessage('user', question);

    // Disable send button
    state.isProcessing = true;
    DOM.sendBtn.disabled = true;
    DOM.sendBtn.innerHTML = '<span>Sending...</span><div class="spinner-border spinner-border-sm" style="width:16px;height:16px;border-width:2px;"></div>';

    // Show typing indicator
    showTyping();

    try {
        const topK = parseInt(DOM.topKSlider.value);
        const temperature = parseFloat(DOM.temperatureSlider.value);
        const includeSources = DOM.includeSources.checked;

        // Use streaming if available
        if (state.isConnected && state.ws) {
            await sendWebSocketQuery(question, topK, temperature, includeSources);
        } else {
            await sendRestQuery(question, topK, temperature, includeSources);
        }

        state.queryCount++;
        updateStats();

    } catch (error) {
        hideTyping();
        addMessage('bot', `❌ Error: ${error.message}`);
        showNotification(`Query failed: ${error.message}`, 'error');
    } finally {
        state.isProcessing = false;
        DOM.sendBtn.disabled = false;
        DOM.sendBtn.innerHTML = '<span>Send</span><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
        DOM.chatInput.focus();
    }
}

async function sendRestQuery(question, topK, temperature, includeSources) {
    const response = await apiRequest('/query', 'POST', {
        question,
        top_k: topK,
        temperature,
        include_sources: includeSources,
        stream: false
    });

    hideTyping();
    displayAnswer(response, includeSources);
}

async function sendWebSocketQuery(question, topK, temperature, includeSources) {
    const message = {
        type: 'query',
        data: {
            query: question,
            settings: {
                top_k: topK,
                temperature: temperature,
                include_sources: includeSources,
                show_thoughts: true
            }
        }
    };

    // Clear any existing stream message
    state.isStreaming = true;
    state.currentStreamMessage = null;

    try {
        state.ws.send(JSON.stringify(message));
    } catch (error) {
        // Fallback to REST if WebSocket fails
        console.warn('WebSocket send failed, falling back to REST');
        await sendRestQuery(question, topK, temperature, includeSources);
    }
}

// ========================================
// WebSocket
// ========================================
function setupWebSocket() {
    const wsUrl = 'ws://localhost:8000/ws/query';

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            state.isConnected = true;
            state.reconnectAttempts = 0;
            updateStatus('connected');
            console.log('🔌 WebSocket connected');
        };

        state.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        state.ws.onclose = () => {
            state.isConnected = false;
            updateStatus('disconnected');
            console.log('🔌 WebSocket disconnected');
            attemptReconnect();
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            // Don't close immediately, let onclose handle it
        };

    } catch (error) {
        console.warn('WebSocket not available:', error);
        updateStatus('disconnected');
        state.isConnected = false;
    }
}

function attemptReconnect() {
    if (state.reconnectAttempts >= state.maxReconnectAttempts) {
        console.log('Max reconnect attempts reached');
        return;
    }

    state.reconnectAttempts++;
    const delay = state.reconnectDelay * Math.pow(1.5, state.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts}/${state.maxReconnectAttempts})`);
    updateStatus('disconnected', `Reconnecting... ${state.reconnectAttempts}/${state.maxReconnectAttempts}`);

    setTimeout(() => {
        setupWebSocket();
    }, delay);
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'welcome':
            addMessage('bot', `👋 ${message.message}`);
            break;

        case 'start':
            console.log('Query started:', message.data);
            break;

        case 'thought':
            // Show thought process
            if (state.currentStreamMessage) {
                // Update existing thought
                const thoughtEl = state.currentStreamMessage.querySelector('.thought-process');
                if (thoughtEl) {
                    thoughtEl.textContent = `🤔 ${message.data.content}`;
                }
            }
            break;

        case 'token':
            // Handle streaming token
            if (!state.currentStreamMessage) {
                // Create new streaming message
                state.currentStreamMessage = addStreamingMessage();
            }
            appendTokenToMessage(state.currentStreamMessage, message.data.content);
            scrollToBottom();
            break;

        case 'source':
            // Handle source update
            if (state.currentStreamMessage) {
                const sourcesEl = state.currentStreamMessage.querySelector('.sources-list');
                if (sourcesEl) {
                    sourcesEl.innerHTML = message.data.sources.map(s =>
                        `<div class="source-item">📚 ${s.text.substring(0, 100)}${s.text.length > 100 ? '...' : ''}</div>`
                    ).join('');
                }
            }
            break;

        case 'progress':
            // Update progress
            const progressEl = document.querySelector('.message.streaming .progress-bar');
            if (progressEl) {
                progressEl.style.width = `${message.data.progress * 100}%`;
            }
            break;

        case 'answer':
            // Final answer
            hideTyping();
            if (state.currentStreamMessage) {
                // Update existing message with final answer
                const contentEl = state.currentStreamMessage.querySelector('.message-content');
                if (contentEl) {
                    contentEl.innerHTML = message.data.answer;
                }
                state.currentStreamMessage.classList.remove('streaming');
            } else {
                // Fallback: create new message
                addMessage('bot', message.data.answer);
            }

            // Add sources if available
            if (message.data.sources && message.data.sources.length > 0) {
                const sourcesHtml = message.data.sources.map(s =>
                    `<div class="source-item">📚 ${s.text.substring(0, 150)}${s.text.length > 150 ? '...' : ''}</div>`
                ).join('');
                addSources(message.data.sources);
            }

            // Reset streaming state
            state.isStreaming = false;
            state.currentStreamMessage = null;
            break;

        case 'done':
            // Stream complete
            state.isStreaming = false;
            if (state.currentStreamMessage) {
                state.currentStreamMessage.classList.remove('streaming');
                state.currentStreamMessage = null;
            }
            break;

        case 'error':
            // Error occurred
            hideTyping();
            addMessage('bot', `❌ ${message.data.message}`);
            state.isStreaming = false;
            state.currentStreamMessage = null;
            showNotification(`Error: ${message.data.message}`, 'error');
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
}

// ========================================
// UI Functions
// ========================================
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';
    const contentHtml = escapeHtml(content).replace(/\n/g, '<br>');

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${contentHtml}</div>
    `;

    DOM.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function addStreamingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot streaming';
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content"></div>
        <div class="thought-process"></div>
        <div class="progress-container">
            <div class="progress-bar" style="width:0%"></div>
        </div>
    `;
    DOM.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function appendTokenToMessage(messageDiv, token) {
    const contentEl = messageDiv.querySelector('.message-content');
    if (contentEl) {
        contentEl.textContent += token;
    }
}

function addSources(sources) {
    if (!sources || sources.length === 0) return;

    const sourceDiv = document.createElement('div');
    sourceDiv.className = 'message bot';
    sourceDiv.innerHTML = `
        <div class="message-avatar">📚</div>
        <div class="message-content" style="font-size:13px;color:var(--text-secondary);">
            <div style="font-weight:600;margin-bottom:4px;">Sources:</div>
            ${sources.map(s =>
                `<div style="margin:2px 0;padding:4px 8px;background:var(--background);border-radius:4px;font-size:12px;">
                    ${s.text.substring(0, 150)}${s.text.length > 150 ? '...' : ''}
                </div>`
            ).join('')}
        </div>
    `;
    DOM.chatMessages.appendChild(sourceDiv);
    scrollToBottom();
}

function displayAnswer(response, includeSources) {
    const content = response.answer || 'No answer generated.';
    addMessage('bot', content);

    if (includeSources && response.sources && response.sources.length > 0) {
        addSources(response.sources);
    }
}

function showTyping() {
    // Remove existing typing indicator
    hideTyping();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    DOM.chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function hideTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

function scrollToBottom() {
    if (DOM.chatMessages) {
        DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    }
}

function autoResizeTextarea() {
    DOM.chatInput.style.height = 'auto';
    DOM.chatInput.style.height = Math.min(DOM.chatInput.scrollHeight, 150) + 'px';
}

function updateCharCount() {
    const count = DOM.chatInput.value.length;
    DOM.charCount.textContent = `${count} characters`;
}

function renderDocumentList() {
    if (state.documents.length === 0) {
        DOM.documentList.innerHTML = '<p class="text-muted">No documents uploaded yet</p>';
        return;
    }

    DOM.documentList.innerHTML = state.documents.map(doc => `
        <div class="document-item">
            <span class="doc-icon">📄</span>
            <span class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
            <span class="doc-size">${formatFileSize(doc.size_bytes)}</span>
            <button class="doc-delete" onclick="deleteDocument('${doc.id}')" title="Delete document">×</button>
        </div>
    `).join('');
}

function renderUploadedFiles(files, result) {
    DOM.uploadedFiles.innerHTML = `
        <h4>Uploaded Files:</h4>
        ${Array.from(files).map(f => `
            <div class="uploaded-file">
                <span>📄 ${escapeHtml(f.name)}</span>
                <span>${formatFileSize(f.size)}</span>
            </div>
        `).join('')}
        ${result.total_chunks > 0 ? `
            <div style="margin-top:8px;padding:8px 12px;background:var(--success);color:white;border-radius:4px;">
                ✅ ${result.total_chunks} chunks created from ${result.total_documents} documents
            </div>
        ` : ''}
    `;
}

function updateStats() {
    DOM.statDocuments.textContent = state.documents.length;
    DOM.statChunks.textContent = state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
    DOM.statQueries.textContent = state.queryCount;
    DOM.docCountHeader.textContent = `${state.documents.length} documents loaded`;
}

function updateUIState() {
    const topK = parseInt(DOM.topKSlider.value);
    const temperature = parseFloat(DOM.temperatureSlider.value);
    document.getElementById('topKValue').textContent = topK;
    document.getElementById('temperatureValue').textContent = temperature.toFixed(1);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function updateStatus(status, message) {
    const dot = DOM.statusBadge.querySelector('.status-dot');
    const text = DOM.statusBadge.querySelector('span:last-child');

    dot.className = 'status-dot';
    if (status === 'connected') {
        dot.classList.add('connected');
        text.textContent = message || 'Connected';
    } else if (status === 'disconnected') {
        dot.classList.add('disconnected');
        text.textContent = message || 'Disconnected';
    } else {
        text.textContent = message || 'Connecting...';
    }
}

function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('docqa-theme', theme);
    document.getElementById('themeToggle').textContent = theme === 'dark' ? '☀️' : '🌙';
}

function toggleTheme() {
    const newTheme = state.theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

// ========================================
// Modal Functions
// ========================================
function openUploadModal() {
    DOM.uploadModal.classList.add('active');
    DOM.fileInput.value = '';
    DOM.uploadedFiles.innerHTML = '';
    DOM.uploadProgress.style.display = 'none';
    DOM.uploadBtn.disabled = false;
    DOM.uploadBtn.textContent = 'Upload';
    DOM.uploadStatus.textContent = '';
}

function closeUploadModal() {
    DOM.uploadModal.classList.remove('active');
}

// ========================================
// File Upload Handlers
// ========================================
function handleDragOver(e) {
    e.preventDefault();
    DOM.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    DOM.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    DOM.uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        DOM.fileInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const files = DOM.fileInput.files;
    if (files.length > 0) {
        // Validate file types
        const validExtensions = ['.pdf', '.docx', '.txt', '.md', '.html', '.csv', '.json'];
        let invalidFiles = [];

        Array.from(files).forEach(f => {
            const ext = '.' + f.name.split('.').pop().toLowerCase();
            if (!validExtensions.includes(ext)) {
                invalidFiles.push(f.name);
            }
        });

        if (invalidFiles.length > 0) {
            showNotification(`Unsupported file types: ${invalidFiles.join(', ')}`, 'warning');
            return;
        }

        // Show selected files
        DOM.uploadedFiles.innerHTML = Array.from(files).map(f => `
            <div class="uploaded-file">
                <span>📄 ${escapeHtml(f.name)}</span>
                <span>${formatFileSize(f.size)}</span>
            </div>
        `).join('');
        DOM.uploadBtn.disabled = false;
    } else {
        DOM.uploadedFiles.innerHTML = '';
        DOM.uploadBtn.disabled = true;
    }
}

// ========================================
// Document Management
// ========================================
async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        await apiRequest(`/documents/${docId}`, 'DELETE');
        await loadDocuments();
        addMessage('bot', '🗑️ Document deleted successfully');
        showNotification('Document deleted successfully', 'success');
    } catch (error) {
        showNotification(`Failed to delete document: ${error.message}`, 'error');
    }
}

async function clearChat() {
    // Remove all messages except the first welcome message
    const messages = DOM.chatMessages.querySelectorAll('.message');
    if (messages.length > 0) {
        // Keep only the first message (welcome)
        messages.forEach((msg, index) => {
            if (index > 0) msg.remove();
        });
    } else {
        // Add a default message
        DOM.chatMessages.innerHTML = `
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <p>Chat cleared. How can I help you today?</p>
                </div>
            </div>
        `;
    }

    state.queryCount = 0;
    state.isStreaming = false;
    state.currentStreamMessage = null;
    updateStats();
    showNotification('Chat cleared', 'info');
}

// ========================================
// Notification System
// ========================================
function showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `notification-toast notification-${type}`;
    toast.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        max-width: 400px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 14px;
        animation: slideUp 0.3s ease;
        background: ${type === 'error' ? 'var(--danger)' :
                   type === 'warning' ? 'var(--warning)' :
                   type === 'success' ? 'var(--success)' : 'var(--primary)'};
        color: white;
    `;

    document.body.appendChild(toast);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) toast.remove();
    }, 5000);
}

// ========================================
// Utility Functions
// ========================================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add CSS for notification animation
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .notification-toast button {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0 4px;
        opacity: 0.7;
    }
    .notification-toast button:hover {
        opacity: 1;
    }

    .message.streaming .progress-container {
        margin-top: 8px;
        height: 3px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
    }
    .message.streaming .progress-bar {
        height: 100%;
        background: var(--primary);
        border-radius: 2px;
        transition: width 0.3s ease;
    }
    .message .thought-process {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 4px;
        font-style: italic;
    }
    .message .sources-list {
        margin-top: 8px;
        font-size: 12px;
        border-top: 1px solid var(--border);
        padding-top: 8px;
    }
    .message .source-item {
        padding: 2px 0;
        color: var(--text-secondary);
    }

    .spinner-border {
        display: inline-block;
        border: 2px solid currentColor;
        border-right-color: transparent;
        border-radius: 50%;
        animation: spinner 0.75s linear infinite;
    }
    @keyframes spinner {
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(styleSheet);

// ========================================
// Debug Helpers
// ========================================
window.debug = {
    state,
    DOM,
    apiRequest,
    loadDocuments,
    sendMessage,
    clearChat,
    uploadFiles,
    toggleTheme
};

console.log('🛠️ Debug helpers available: window.debug');
