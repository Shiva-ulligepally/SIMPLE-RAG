import React, { useState, useEffect, useRef } from 'react';
import API_BASE from './config/api';

function App() {
  // Config & API States
  const [groqApiKey, setGroqApiKey] = useState(() => localStorage.getItem('groq_api_key') || '');
  const [chromaUrl, setChromaUrl] = useState(() => localStorage.getItem('chroma_url') || 'http://localhost:8000');
  const [chromaApiKey, setChromaApiKey] = useState(() => localStorage.getItem('chroma_api_key') || '');
  
  // RAG Parameters
  const [searchType, setSearchType] = useState('similarity'); // 'similarity' | 'keyword' | 'chroma'
  const [chunkSize, setChunkSize] = useState(150);
  const [chunkOverlap, setChunkOverlap] = useState(30);
  const [numResults, setNumResults] = useState(3);
  
  // App Operational States
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Welcome to AI assistance that helps to understand about the uploaded documents.',
      meta: { engine: 'system' }
    }
  ]);
  const [activeInspect, setActiveInspect] = useState(null); // stores inspector info of currently selected message
  const [serverStatus, setServerStatus] = useState({
    online: false,
    pdfFound: false,
    isIndexed: false,
    indexStats: null,
    chroma: { connected: false, url: 'http://localhost:8000', collection: '' }
  });
  
  // Loading & Alert UI States
  const [isIndexing, setIsIndexing] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [alert, setAlert] = useState(null);

  const messagesEndRef = useRef(null);

  // Auto-save credentials to localStorage
  useEffect(() => {
    localStorage.setItem('groq_api_key', groqApiKey);
  }, [groqApiKey]);

  useEffect(() => {
    localStorage.setItem('chroma_url', chromaUrl);
  }, [chromaUrl]);

  useEffect(() => {
    localStorage.setItem('chroma_api_key', chromaApiKey);
  }, [chromaApiKey]);

  // Fetch status from backend on mount and when connection params change
  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status?chromaUrl=${encodeURIComponent(chromaUrl)}&chromaApiKey=${encodeURIComponent(chromaApiKey)}`);
      if (res.ok) {
        const data = await res.json();
        setServerStatus({
          online: true,
          pdfFound: data.pdfFound,
          isIndexed: data.isIndexed,
          indexStats: data.indexStats,
          chroma: data.chroma
        });
      } else {
        throw new Error('Server not online');
      }
    } catch (e) {
      setServerStatus(prev => ({ ...prev, online: false }));
    }
  };

  useEffect(() => {
    fetchStatus();
    // Poll status every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, [chromaUrl, chromaApiKey]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isQuerying]);

  // Trigger server PDF parsing and index creation
  const handleIndex = async () => {
    setIsIndexing(true);
    setAlert(null);
    try {
      const response = await fetch(`${API_BASE}/api/index-existing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunkSize,
          chunkOverlap,
          chromaUrl,
          chromaApiKey
        })
      });
      const data = await response.json();
      if (data.success) {
        setAlert({ type: 'success', message: 'Document split and vector database loaded successfully!' });
        setServerStatus(prev => ({
          ...prev,
          isIndexed: true,
          indexStats: data.stats,
          chroma: { ...prev.chroma, connected: data.chroma.indexed }
        }));
      } else {
        setAlert({ type: 'error', message: data.error || 'Indexing failed.' });
      }
    } catch (err) {
      setAlert({ type: 'error', message: `Could not connect to Node.js backend: ${err.message}` });
    } finally {
      setIsIndexing(false);
    }
  };

  // Submit Query to RAG backend
  const handleSubmitQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    if (isQuerying) return;



    const currentQuery = query;
    setQuery('');
    setAlert(null);

    // 1. Add User query to history
    const userMessageId = `user_${Date.now()}`;
    const botMessageId = `bot_${Date.now()}`;
    
    setChatHistory(prev => [
      ...prev,
      {
        id: userMessageId,
        role: 'user',
        content: currentQuery,
        meta: { timestamp: new Date().toLocaleTimeString() }
      }
    ]);

    setIsQuerying(true);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          searchType,
          numResults,
          chunkSize,
          chunkOverlap,
          groqApiKey,
          chromaUrl,
          chromaApiKey
        })
      });

      const data = await response.json();

      if (data.success) {
        // 2. Add LLM Answer to history
        setChatHistory(prev => [
          ...prev,
          {
            id: botMessageId,
            role: 'assistant',
            content: data.answer,
            isUnrelated: data.isUnrelated,
            meta: {
              timestamp: new Date().toLocaleTimeString(),
              engine: data.usedEngine,
              chunksCount: data.retrievedChunks?.length || 0
            },
            inspector: {
              retrievedChunks: data.retrievedChunks || [],
              systemPrompt: data.inspector?.systemPrompt || '',
              userQuery: data.inspector?.userQuery || ''
            }
          }
        ]);

        // Auto-select latest bot response for RAG Inspector
        setActiveInspect({
          retrievedChunks: data.retrievedChunks || [],
          systemPrompt: data.inspector?.systemPrompt || '',
          userQuery: data.inspector?.userQuery || ''
        });

        // Trigger indexing status update in case auto-indexing happened
        fetchStatus();
      } else {
        setChatHistory(prev => [
          ...prev,
          {
            id: botMessageId,
            role: 'assistant',
            content: `⚠️ Error from server: ${data.error || 'Failed to generate answer.'}`,
            meta: { engine: 'system' }
          }
        ]);
      }
    } catch (err) {
      setChatHistory(prev => [
        ...prev,
        {
          id: botMessageId,
          role: 'assistant',
          content: `❌ Connection error to RAG backend: ${err.message}. Make sure Node backend is running.`,
          meta: { engine: 'system' }
        }
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="app-container">
      {/* SIDEBAR: Configuration Panel */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <div className="brand-icon">R</div>
          <div>
            <h1 className="brand-name">Naive RAG</h1>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>10-Yr Architect Standard</p>
          </div>
        </div>

        {/* Server Status Health check */}
        <div className="glass-card status-widget">
          <div className="status-row">
            <span>RAG Backend:</span>
            <span className="status-value">
              <span className={`status-dot ${serverStatus.online ? 'active' : 'inactive'}`}></span>
              {serverStatus.online ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="status-row" style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-start' }}>
            <span>Document(s):</span>
            <span className="status-value" style={{ fontSize: '11px', color: serverStatus.pdfFound ? 'var(--accent-success)' : 'var(--accent-danger)', wordBreak: 'break-all', textAlign: 'left' }}>
              {serverStatus.pdfFound 
                ? (serverStatus.indexStats ? serverStatus.indexStats.fileName : (serverStatus.pdfFiles ? serverStatus.pdfFiles.join(', ') : 'PDFs Detected')) 
                : 'Missing PDF'}
            </span>
          </div>
          <div className="status-row">
            <span>Index Status:</span>
            <span className="status-value" style={{ fontWeight: 600 }}>
              {serverStatus.isIndexed ? 'Indexed ✅' : 'Not Indexed ❌'}
            </span>
          </div>
          {serverStatus.isIndexed && serverStatus.indexStats && (
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '6px', marginTop: '4px' }}>
              <div>Chunks: {serverStatus.indexStats.chunkCount} (Size: {serverStatus.indexStats.chunkSize} words)</div>
              <div>Vocab Size: {serverStatus.indexStats.vocabularySize} words</div>
            </div>
          )}
        </div>



        {/* RAG Tuning Panel */}
        <div>
          <h2 className="section-title">RAG Parameters</h2>
          <div className="form-group">
            <label className="form-label">Search Strategy</label>
            <select
              className="form-select"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
            >
              <option value="similarity">Similarity Search (Local TF-IDF)</option>
              <option value="keyword">Keyword Search (Local Term Match)</option>
              <option value="chroma">Similarity Search (Chroma DB)</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">
              <span>Chunk Size (Words)</span>
              <span style={{ color: 'var(--accent-primary)' }}>{chunkSize}</span>
            </label>
            <input
              type="range"
              min="50"
              max="400"
              step="10"
              value={chunkSize}
              onChange={(e) => setChunkSize(parseInt(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              <span>Chunk Overlap (Words)</span>
              <span style={{ color: 'var(--accent-primary)' }}>{chunkOverlap}</span>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              <span>Retrieve Count (Top-K)</span>
              <span style={{ color: 'var(--accent-primary)' }}>{numResults}</span>
            </label>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={numResults}
              onChange={(e) => setNumResults(parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* Index Action Button */}
        <button
          className={`btn btn-primary ${isIndexing || !serverStatus.pdfFound ? 'btn-disabled' : ''}`}
          onClick={handleIndex}
          disabled={isIndexing || !serverStatus.pdfFound}
        >
          {isIndexing ? (
            <>
              <span className="loading-spinner"></span>
              Indexing...
            </>
          ) : (
            'Process & Index PDF'
          )}
        </button>

        {alert && (
          <div className={`alert-banner ${alert.type}`}>
            {alert.type === 'error' ? '⚠️' : '✅'} {alert.message}
          </div>
        )}
      </aside>

      {/* WORKSPACE Split: Chat + Inspector */}
      <main className="main-workspace">
        {/* Chat Section */}
        <div className="chat-container glass-panel">
          <header className="chat-header">
            <div className="chat-header-title">
              <h2>Simple RAG Chat Room</h2>
              <p>Strict anti-hallucination context validation</p>
            </div>
            {serverStatus.chroma.connected && (
              <span className="inspect-badge" style={{ backgroundColor: 'rgba(16,185,129,0.1)', color: 'var(--accent-success)' }}>
                Chroma DB Integrated
              </span>
            )}
          </header>

          {/* Messages feed */}
          <div className="chat-messages">
            {chatHistory.map((msg) => (
              <div
                key={msg.id}
                className={`message ${msg.role === 'user' ? 'message-user' : 'message-bot'} ${msg.isUnrelated ? 'unrelated' : ''}`}
              >
                {msg.role === 'assistant' && (
                  <div className="message-source-tag">
                    {msg.isUnrelated ? '⚠️ UNRELATED WARNING' : '📄 CONTEXT BOUND ANSWER'}
                  </div>
                )}
                <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                {msg.meta && (
                  <div className="message-meta">
                    <span>Engine: {msg.meta.engine || 'N/A'}</span>
                    <span style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      {msg.inspector && (
                        <span
                          className="inspect-badge"
                          onClick={() => setActiveInspect(msg.inspector)}
                        >
                          Inspect RAG
                        </span>
                      )}
                      <span>{msg.meta.timestamp}</span>
                    </span>
                  </div>
                )}
              </div>
            ))}

            {isQuerying && (
              <div className="message message-bot" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className="loading-spinner" style={{ borderColor: 'rgba(99, 102, 241, 0.3)', borderTopColor: 'var(--accent-secondary)' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Searching vector space, injecting context and generating strict completion...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Query Form Input */}
          <form onSubmit={handleSubmitQuery} className="chat-input-area">
            <input
              className="chat-input"
              placeholder="Ask a question about Fundamental Rights (e.g. What is Article 14?)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isQuerying}
            />
            <button
              type="submit"
              className={`btn btn-primary ${isQuerying || !query.trim() ? 'btn-disabled' : ''}`}
              disabled={isQuerying || !query.trim()}
              style={{ width: '100px', height: '56px', borderRadius: '12px' }}
            >
              Ask
            </button>
          </form>
        </div>

        {/* RAG Inspector Panel */}
        <div className="inspector-panel glass-panel">
          <header className="inspector-header">
            <h2 className="inspector-title">
              <span className="inspector-pulse"></span>
              RAG Pipeline Inspector
            </h2>
          </header>

          {activeInspect ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <h3 className="section-title" style={{ marginTop: 0 }}>Active Prompt Query</h3>
                <div style={{ fontSize: '14px', padding: '10px', background: 'var(--bg-secondary)', borderRadius: '6px', border: '1px solid var(--glass-border)' }}>
                  "{activeInspect.userQuery}"
                </div>
              </div>

              <div>
                <h3 className="section-title">Retrieved Chunks ({activeInspect.retrievedChunks.length})</h3>
                {activeInspect.retrievedChunks.map((res, i) => (
                  <div key={res.chunk?.id || i} className="glass-card chunk-card">
                    <div className="chunk-header">
                      <span className="chunk-id">#{res.chunk?.id || `chunk_${i}`}</span>
                      {res.score !== undefined && (
                        <span className="chunk-score">
                          Score: {res.score.toFixed(4)}
                        </span>
                      )}
                    </div>
                    <div className="chunk-text">
                      {res.chunk?.text}
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <h3 className="section-title">Full Injected System Prompt</h3>
                <div className="prompt-box">
                  {activeInspect.systemPrompt}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-chat">
              <div className="empty-chat-icon">🔍</div>
              <p style={{ fontSize: '14px' }}>No active inspect session.</p>
              <p style={{ fontSize: '12px', padding: '0 20px' }}>Ask a question or click "Inspect RAG" on any answer in the chat logs to see full context injection logs.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
