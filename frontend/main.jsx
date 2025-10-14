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
      const res = await fetch("/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
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
