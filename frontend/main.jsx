const { StrictMode, useEffect, useMemo, useRef, useState } = React;
const { createRoot } = ReactDOM;

function App() {
  const [repoName, setRepoName] = useState(window.REPO_NAME || null);
  const [repoUrl, setRepoUrl] = useState("");
  const [messages, setMessages] = useState([
    { id: 1, role: "assistant", content: "Ask me about your codebase. I'll search the vector store." },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [starCount, setStarCount] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  useEffect(() => {
    const fetchStarCount = async () => {
      try {
        const response = await fetch("https://api.github.com/repos/ValerianRey/RagThisCode");
        if (response.ok) {
          const data = await response.json();
          setStarCount(data.stargazers_count);
        }
      } catch (e) {
        // Silently fail if we can't fetch star count
      }
    };
    fetchStarCount();
  }, []);

  const handleSend = async (e) => {
    e?.preventDefault?.();
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput("");

    const user = { id: Date.now(), role: "user", content: text };
    setMessages((prev) => [...prev, user]);

    const assistantId = Date.now() + 1;
    const placeholder = { id: assistantId, role: "assistant", content: "Thinking..." };
    setMessages((prev) => [...prev, placeholder]);

    try {
      const res = await fetch("/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text , repo_name: repoName}),
      });

      if (!res.body) {
        throw new Error("No response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = value ? decoder.decode(value, { stream: !done }) : "";
        if (chunk) {
          acc += chunk;
          const parts = acc.split("\n[DONE]");
          const toRender = parts[0];
          setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: toRender } : m)));
          if (parts.length > 1) {
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: toRender, final: true } : m)));
            break;
          }
        }
      }
    } catch (e) {
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: "Error contacting backend.", final: true } : m)));
    }

    setSending(false);
  };

  const handleRepoSubmit = (e) => {
    e?.preventDefault?.();
    const url = repoUrl.trim();
    if (!url) return;

    let repoName = "";
    try {
      const urlObj = new URL(url);
      if (urlObj.hostname === "github.com" || urlObj.hostname === "www.github.com") {
        const pathParts = urlObj.pathname.split("/").filter(p => p);
        if (pathParts.length >= 2) {
          repoName = `${pathParts[0]}/${pathParts[1]}`;
        }
      }
    } catch (e) {
      if (url.includes("/")) {
        repoName = url.replace(/^https?:\/\//, "").replace(/^github\.com\//, "").replace(/^www\.github\.com\//, "");
      } else {
        return;
      }
    }

    if (repoName) {
      window.location.href = `/${repoName}`;
    }
  };

  if (!repoName) {
    return (
      <div className="container">
        <div className="topbar">
          <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 1 }}>
            <a href="/">
              <img src="/assets/logo.png" alt="RagThisCode" className="logo" />
            </a>
            <div>
              <div className="title">RagThisCode</div>
              <div className="subtitle">Code-aware chat powered by an agentic RAG system over your codebase.</div>
            </div>
          </div>
          <a
            href="https://github.com/ValerianRey/RagThisCode"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            <svg className="github-octocat" width="20" height="20" viewBox="0 0 16 16" fill="#24292f">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            <span className="github-text">GitHub</span>
            <svg className="github-star" width="16" height="16" viewBox="0 0 16 16" fill="#F9D71C">
              <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"/>
            </svg>
            {starCount !== null && <span className="star-count">{starCount >= 1000 ? (starCount / 1000).toFixed(1) + 'k' : starCount}</span>}
          </a>
        </div>

        <div className="messages" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ maxWidth: "600px", width: "100%", padding: "40px 20px" }}>
            <h2 style={{ marginBottom: "20px", textAlign: "center" }}>Enter a GitHub Repository</h2>
            <p style={{ color: "var(--muted)", marginBottom: "30px", textAlign: "center" }}>
              Paste a GitHub repository URL to get started
            </p>
            <form onSubmit={handleRepoSubmit} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <input
                className="input"
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                style={{ fontSize: "16px", padding: "14px" }}
              />
              <button
                className="send"
                type="submit"
                disabled={!repoUrl.trim()}
                style={{ width: "100%", fontSize: "16px", padding: "14px" }}
              >
                Load Repository
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="topbar">
        <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 1 }}>
          <a href="/">
            <img src="/assets/logo.png" alt="RagThisCode" className="logo" />
          </a>
          <div>
            <div className="title">RagThisCode</div>
            <div className="subtitle">
              {repoName ? <>You are chatting about the repository <strong><a href={`https://github.com/${repoName}`} target="_blank" style={{ color: "inherit", textDecoration: "underline" }}>{repoName}</a></strong>.</> : "Code-aware chat powered by an agentic RAG system over your codebase."}
            </div>
          </div>
        </div>
        <a
          href="https://github.com/ValerianRey/RagThisCode"
          target="_blank"
          rel="noopener noreferrer"
          className="github-link"
        >
          <svg className="github-octocat" width="20" height="20" viewBox="0 0 16 16" fill="#24292f">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
          </svg>
          <span className="github-text">GitHub</span>
          <svg className="github-star" width="16" height="16" viewBox="0 0 16 16" fill="#F9D71C">
            <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Zm0 2.445L6.615 5.5a.75.75 0 0 1-.564.41l-3.097.45 2.24 2.184a.75.75 0 0 1 .216.664l-.528 3.084 2.769-1.456a.75.75 0 0 1 .698 0l2.77 1.456-.53-3.084a.75.75 0 0 1 .216-.664l2.24-2.183-3.096-.45a.75.75 0 0 1-.564-.41L8 2.694Z"/>
          </svg>
          {starCount !== null && <span className="star-count">{starCount >= 1000 ? (starCount / 1000).toFixed(1) + 'k' : starCount}</span>}
        </a>
      </div>

      <div className="messages">
        <div className="hint">Tip: Ask things like "Explain the implementation of XYZ"</div>
        {messages.map((m) => (
          <div key={m.id} className={`bubble ${m.role}`}>
            <div className="avatar">{m.role === "assistant" ? "🤖" : "👤"}</div>
            <div>
              <div className="role">{m.role}</div>
              <div className="content" style={m.final ? { borderColor: "#7c9cff", boxShadow: "0 0 0 1px rgba(124,156,255,0.25) inset" } : undefined}>
                <div dangerouslySetInnerHTML={{ __html: marked.parse(m.content) }} />
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form className="composer" onSubmit={handleSend}>
        <input
          className="input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your codebase..."
        />
        <button className="send" disabled={sending || !input.trim()} onClick={handleSend}>
          {sending ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(
  <StrictMode>
    <App />
  </StrictMode>
);
