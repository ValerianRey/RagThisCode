const { StrictMode, useEffect, useMemo, useRef, useState } = React;
const { createRoot } = ReactDOM;

function App() {
  const [messages, setMessages] = useState([
    { id: 1, role: "assistant", content: "Ask me about your codebase. I'll search the vector store." },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

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
      const res = await fetch("http://127.0.0.1:7070/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      const finalText = typeof data.final === "string" ? data.final : String(data.final ?? "");
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: `Final answer:\n${finalText}`, final: true } : m)));
    } catch (e) {
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: "Error contacting backend.", final: true } : m)));
    }

    setSending(false);
  };

  return (
    <div className="container">
      <div className="topbar">
        <div className="logo" />
        <div>
          <div className="title">RagThisCode</div>
          <div className="subtitle">Code-aware chat powered by your vector store</div>
        </div>
      </div>

      <div className="messages">
        <div className="hint">Tip: Ask things like "Explain UPGrad implementation"</div>
        {messages.map((m) => (
          <div key={m.id} className={`bubble ${m.role}`}>
            <div className="avatar">{m.role === "assistant" ? "🤖" : "👤"}</div>
            <div>
              <div className="role">{m.final ? "assistant (final)" : m.role}</div>
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
