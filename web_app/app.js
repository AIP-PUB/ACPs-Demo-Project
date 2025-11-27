// Basic front-end logic for Tour Assistant
// Assumptions: backend FastAPI server runs on same host:port as served static files or accessible via relative path
// API endpoint: POST /user_api { session_id?: string, query: string }
// Response: { session_id, analysis, partner_results, partner_subqueries, final_response }

const els = {
    messages: document.getElementById('messages'),
    analysisJson: document.getElementById('analysisJson'),
    partnerResults: document.getElementById('partnerResults'),
    finalResponse: document.getElementById('finalResponse'),
    userInput: document.getElementById('userInput'),
    form: document.getElementById('inputForm'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    sessionDisplay: document.getElementById('sessionDisplay'),
};

// Backend base logic: 如需修改地址，可在 config.js 中设置 window.APP_CONFIG.backendBase。
const CONFIG = window.APP_CONFIG || {};
const DEFAULT_BACKEND_BASE = 'http://127.0.0.1:8019';
const runtimeBackendBase = typeof CONFIG.backendBase === 'string' ? CONFIG.backendBase.trim() : null;
const BACKEND_BASE = runtimeBackendBase === '' ? '' : runtimeBackendBase || DEFAULT_BACKEND_BASE;
const BACKEND_BASE_PREFIX = BACKEND_BASE.endsWith('/') ? BACKEND_BASE.slice(0, -1) : BACKEND_BASE;
function apiUrl(path) {
    // 若本身就运行在 8019 或由 tour_assistant.py 挂载 /web 下，则直接用相对路径
    if (location.port === '8019' || BACKEND_BASE === '') return path;
    return BACKEND_BASE_PREFIX + path;
}

let currentSessionId = null; // will be set after first response
let isSending = false;

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    els.messages.appendChild(div);
    els.messages.scrollTop = els.messages.scrollHeight;
}

function setSessionId(id) {
    currentSessionId = id;
    els.sessionDisplay.textContent = id ? `Session: ${id}` : '';
}

function resetSessionUI() {
    setSessionId(null);
    els.messages.innerHTML = '';
    els.analysisJson.textContent = '等待发送第一条消息...';
    els.analysisJson.classList.add('placeholder');
    els.partnerResults.innerHTML = '暂无结果';
    els.partnerResults.classList.add('placeholder');
    els.finalResponse.textContent = '暂无结果';
    els.finalResponse.classList.add('placeholder');
}

function pretty(obj) {
    return JSON.stringify(obj, null, 2);
}

function renderAnalysis(analysis) {
    els.analysisJson.textContent = pretty(analysis);
    els.analysisJson.classList.remove('placeholder');
}

function renderPartnerResults(results, subqueries) {
    if (!results || Object.keys(results).length === 0) {
        els.partnerResults.innerHTML = '无代理参与';
        return;
    }
    els.partnerResults.classList.remove('placeholder');
    els.partnerResults.innerHTML = '';
    Object.entries(results).forEach(([agentId, data]) => {
        const card = document.createElement('div');
        card.className = 'partner-card';
        const title = document.createElement('h3');
        const state = data.state || '-';
        title.innerHTML = `<span>${agentId}</span><code style="font-size:11px; color:#555">${state}</code>`;
        const subQ = subqueries && subqueries[agentId] ? subqueries[agentId] : '(未提供)';
        const content = document.createElement('pre');
        const product = data.product_text || data.error || '(无产出)';
        content.textContent = `[子查询]\n${subQ}\n\n[产出]\n${product}`;
        card.appendChild(title);
        card.appendChild(content);
        els.partnerResults.appendChild(card);
    });
}

function renderFinalResponse(text) {
    els.finalResponse.classList.remove('placeholder');
    // Basic Markdown-ish -> HTML (very light)
    const safe = text
        .replace(/</g, '&lt;')
        .replace(/^# (.*)$/gm, '<h2>$1</h2>')
        .replace(/^## (.*)$/gm, '<h3>$1</h3>')
        .replace(/^\* (.*)$/gm, '<li>$1</li>')
        .replace(/\n\n/g, '<br/><br/>');
    // Wrap lone <li> groups with <ul>
    const wrapped = safe.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    els.finalResponse.innerHTML = wrapped;
}

async function sendQuery(query) {
    if (!query.trim()) return;
    if (isSending) return;
    isSending = true;
    els.form.querySelector('button[type="submit"]').disabled = true;
    addMessage('user', query);
    try {
        const payload = { query };
        if (currentSessionId) payload.session_id = currentSessionId;
        const resp = await fetch(apiUrl('/user_api'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${text}`);
        }
        const data = await resp.json();
        if (!currentSessionId) setSessionId(data.session_id);
        renderAnalysis(data.analysis);
        renderPartnerResults(data.partner_results, data.partner_subqueries);
        renderFinalResponse(data.final_response);
        addMessage('assistant', data.final_response);
    } catch (err) {
        console.error(err);
        addMessage('assistant', `发生错误：${err.message}`);
    } finally {
        isSending = false;
        els.form.querySelector('button[type="submit"]').disabled = false;
    }
}

els.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = els.userInput.value;
    els.userInput.value = '';
    sendQuery(query);
});

els.newSessionBtn.addEventListener('click', () => {
    resetSessionUI();
    addMessage('system', '已创建新的会话，请输入您的需求。');
});

// Initial state
resetSessionUI();
addMessage('system', '欢迎！输入您的旅行需求开始对话。');
