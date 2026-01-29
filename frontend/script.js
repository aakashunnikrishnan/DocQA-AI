/**
 * DocQA AI - Frontend JavaScript
 * Handles chat, document upload, and API interactions
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
    theme: 'light'
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
    temperatureSlider: document.getElementById('temperatureSlider')
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

    // Settings
    DOM.includeSources.addEventListener('change', () => {});

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+K or Cmd+K to focus chat input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            DOM.chatInput.focus();
        }
        // Escape to close modal
        if (e.key === 'Escape') {
            closeUploadModal();
        }
    });
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
        },
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }
        return await response.json();
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
        } else {
            updateStatus('disconnected');
            console.warn('⚠️ API is not responding');
        }
    } catch (error) {
        updateStatus('disconnected');
        console.warn('⚠️ Cannot connect to API:', error.message);
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
        DOM.documentList.innerHTML = '<p class="text-muted">Error loading documents</p>';
    }
}

async function uploadFiles() {
    const files = DOM.fileInput.files;
    if (files.length === 0) {
        alert('Please select files to upload');
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

    try {
        const response = await fetch(`${state.apiBase}/documents/ingest`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();

        DOM.uploadProgressFill.style.width = '100%';
        DOM.uploadStatus.textContent = '✅ Upload complete!';

        // Show uploaded files
        renderUploadedFiles(files, result);

        // Reload documents
        await loadDocuments();

        // Close modal after delay
        setTimeout(() => {
            closeUploadModal();
            DOM.uploadProgress.style.display = 'none';
            DOM.uploadBtn.disabled = false;
        }, 2000);

        // Show success message in chat
        addMessage('bot', `✅ Successfully ingested ${result.total_documents} documents (${result.total_chunks} chunks)`);

    } catch (error) {
        DOM.uploadStatus.textContent = `❌ Error: ${error.message}`;
        DOM.uploadBtn.disabled = false;
        console.error('Upload error:', error);
    }
}

async function sendMessage() {
    const question = DOM.chatInput.value.trim();
    if (!question || state.isProcessing) return;

    // Add user message
    addMessage('user', question);
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    updateCharCount();

    // Show typing indicator
    showTyping();
    state.isProcessing = true;
    DOM.sendBtn.disabled = true;

    try {
        const topK = parseInt(DOM.topKSlider.value);
        const temperature = parseFloat(DOM.temperatureSlider.value);
        const includeSources = DOM.includeSources.checked;

        const response = await apiRequest('/query', 'POST', {
            question,
            top_k: topK,
            temperature,
            include_sources: includeSources,
            stream: false
        });

        hideTyping();

        // Format sources
        let sourcesHtml = '';
        if (includeSources && response.sources && response.sources.length > 0) {
            sourcesHtml = '<div class="source-citation">📚 Sources:<br>';
            response.sources.forEach((source, i) => {
                sourcesHtml += `<span style="font-size:13px;color:var(--text-secondary);">
                    ${i+1}. Score: ${(source.score * 100).toFixed(1)}%
                    ${source.metadata?.file_path ? `- ${source.metadata.file_path.split('/').pop()}` : ''}
                </span><br>`;
            });
            sourcesHtml += '</div>';
        }

        // Build confidence badge
        let confidenceHtml = '';
        if (response.confidence !== undefined) {
            const percent = (response.confidence * 100).toFixed(0);
            let cls = '';
            if (percent < 50) cls = 'very-low';
            else if (percent < 70) cls = 'low';
            confidenceHtml = `<span class="confidence-badge ${cls}">${percent}% confidence</span>`;
        }

        // Add bot message with sources
        const content = response.answer || 'No answer generated.';
        addMessage('bot', content + (response.has_hallucination ? '\n\n⚠️ *This response may contain unsupported information.*' : ''));

        // Add confidence and sources separately
        if (confidenceHtml || sourcesHtml) {
            const sourceMsg = document.createElement('div');
            sourceMsg.className = 'message bot';
            sourceMsg.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content" style="font-size:13px;">
                    ${confidenceHtml}
                    ${sourcesHtml}
                </div>
            `;
            DOM.chatMessages.appendChild(sourceMsg);
            scrollToBottom();
        }

        // Update query count
        state.queryCount++;
        updateStats();

    } catch (error) {
        hideTyping();
        addMessage('bot', `❌ Error: ${error.message}`);
    } finally {
        state.isProcessing = false;
        DOM.sendBtn.disabled = false;
        DOM.chatInput.focus();
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
            updateStatus('connected');
            console.log('🔌 WebSocket connected');
        };

        state.ws.onmessage = (event) => {
            handleWebSocketMessage(JSON.parse(event.data));
        };

        state.ws.onclose = () => {
            state.isConnected = false;
            updateStatus('disconnected');
            console.log('🔌 WebSocket disconnected');
            // Reconnect after 5 seconds
            setTimeout(setupWebSocket, 5000);
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

    } catch (error) {
        console.warn('WebSocket not available:', error);
        updateStatus('disconnected');
    }
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'welcome':
            addMessage('bot', `👋 ${message.message}`);
            break;

        case 'stream':
            // Handle streaming chunks
            const lastMsg = DOM.chatMessages.lastElementChild;
            if (lastMsg && lastMsg.classList.contains('streaming')) {
                const content = lastMsg.querySelector('.message-content');
                content.textContent += message.chunk;
                scrollToBottom();
            } else {
                // Start new streaming message
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message bot streaming';
                msgDiv.innerHTML = `
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">${message.chunk}</div>
                `;
                DOM.chatMessages.appendChild(msgDiv);
                scrollToBottom();
            }
            break;

        case 'answer':
            // Remove streaming class and finalize
            const streamingMsg = DOM.chatMessages.querySelector('.streaming');
            if (streamingMsg) {
                streamingMsg.classList.remove('streaming');
                const content = streamingMsg.querySelector('.message-content');
                // Add sources if available
                if (message.sources && message.sources.length > 0) {
                    let sourcesHtml = '<div class="source-citation">📚 Sources:<br>';
                    message.sources.forEach((source, i) => {
                        sourcesHtml += `<span style="font-size:13px;color:var(--text-secondary);">
                            ${i+1}. Score: ${(source.score * 100).toFixed(1)}%
                        </span><br>`;
                    });
                    sourcesHtml += '</div>';
                    content.innerHTML += sourcesHtml;
                }
                scrollToBottom();
            }
            break;

        case 'status':
            // Update status message
            break;

        case 'error':
            addMessage('bot', `❌ ${message.message}`);
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
    const contentHtml = content.replace(/\n/g, '<br>');

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${contentHtml}</div>
    `;

    DOM.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function showTyping() {
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
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
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
            <span class="doc-name" title="${doc.name}">${doc.name}</span>
            <span class="doc-size">${formatFileSize(doc.size_bytes)}</span>
            <button class="doc-delete" onclick="deleteDocument('${doc.id}')" title="Delete document">×</button>
        </div>
    `).join('');
}

function renderUploadedFiles(files, result) {
    DOM.uploadedFiles.innerHTML = files.length > 0 ? `
        <h4>Uploaded Files:</h4>
        ${Array.from(files).map(f => `
            <div class="uploaded-file">
                <span>📄 ${f.name}</span>
                <span>${formatFileSize(f.size)}</span>
            </div>
        `).join('')}
        ${result.total_chunks > 0 ? `
            <div style="margin-top:8px;padding:8px 12px;background:var(--success);color:white;border-radius:4px;">
                ✅ ${result.total_chunks} chunks created from ${result.total_documents} documents
            </div>
        ` : ''}
    ` : '';
}

function updateStats() {
    DOM.statDocuments.textContent = state.documents.length;
    DOM.statChunks.textContent = state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
    DOM.statQueries.textContent = state.queryCount;
    DOM.docCountHeader.textContent = `${state.documents.length} documents loaded`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function updateStatus(status) {
    const dot = DOM.statusBadge.querySelector('.status-dot');
    const text = DOM.statusBadge.querySelector('span:last-child');

    dot.className = 'status-dot';
    if (status === 'connected') {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else if (status === 'disconnected') {
        dot.classList.add('disconnected');
        text.textContent = 'Disconnected';
    } else {
        text.textContent = 'Connecting...';
    }
}

function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('docqa-theme', theme);
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
}

function closeUploadModal() {
    DOM.uploadModal.classList.remove('active');
}

// Handle click outside modal to close
DOM.uploadModal.addEventListener('click', (e) => {
    if (e.target === DOM.uploadModal) {
        closeUploadModal();
    }
});

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
        // Show selected files
        DOM.uploadedFiles.innerHTML = Array.from(files).map(f => `
            <div class="uploaded-file">
                <span>📄 ${f.name}</span>
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
        addMessage('bot', `🗑️ Document deleted successfully`);
    } catch (error) {
        alert(`Failed to delete document: ${error.message}`);
    }
}

async function clearChat() {
    DOM.chatMessages.innerHTML = `
        <div class="message bot">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <p>Chat cleared. How can I help you today?</p>
            </div>
        </div>
    `;
    state.queryCount = 0;
    updateStats();
}

// ========================================
// Keyboard Shortcuts
// ========================================
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to send
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendMessage();
    }
});

// ========================================
// Export for debugging
// ========================================
window.debug = {
    state,
    DOM,
    apiRequest,
    loadDocuments,
    sendMessage,
    clearChat
};
