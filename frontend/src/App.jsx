import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, Bot, User, Loader2, ArrowUp } from "lucide-react";

const API_URL = "http://localhost:5000";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([
    {
      type: "bot",
      content: {
        status: "success",
        task_title: "System Ready",
        steps: [
          {
            step: 0,
            instruction: "Repair Assistant Online. What do you need help with?",
          },
        ],
      },
    },
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!query.trim()) return;

    // Add User Message
    const newMsgs = [...messages, { type: "user", content: query }];
    setMessages(newMsgs);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/api/chat`, { query: query });
      setMessages((prev) => [...prev, { type: "bot", content: res.data }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: {
            status: "error",
            message: "Connection failure.",
          },
        },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-white text-black font-sans selection:bg-black selection:text-white">
      {/* Header */}
      <header className="border-b border-black p-5 flex items-center justify-between sticky top-0 bg-white/90 backdrop-blur-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-black text-white flex items-center justify-center rounded-full">
            <Bot size={18} />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">PRISM REPAIR</h1>
        </div>
        <div className="text-xs font-mono text-gray-400">V 1.0</div>
      </header>

      {/* Chat Container */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-8 max-w-3xl mx-auto w-full">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${
              msg.type === "user" ? "items-end" : "items-start"
            }`}
          >
            {/* Message Bubble */}
            <div
              className={`max-w-[90%] sm:max-w-[80%] rounded-none p-0 ${
                msg.type === "user" ? "text-right" : "text-left w-full"
              }`}
            >
              {typeof msg.content === "string" ? (
                <p className="text-lg font-light leading-snug border-b border-black pb-2 inline-block">
                  {msg.content}
                </p>
              ) : (
                /* Structured JSON Response */
                <StructuredResponse data={msg.content} />
              )}
            </div>

            {/* Timestamp / Role Label */}
            <span className="text-[10px] uppercase tracking-widest text-gray-400 mt-2 font-mono">
              {msg.type === "user" ? "User Input" : "System Response"}
            </span>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-gray-400 font-mono animate-pulse">
            <Loader2 size={14} className="animate-spin" /> PROCESSING...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-6 bg-white border-t border-black">
        <div className="max-w-3xl mx-auto flex items-center gap-4">
          <input
            type="text"
            className="flex-1 bg-transparent border-b border-gray-300 py-3 text-lg focus:outline-none focus:border-black transition-colors placeholder:text-gray-300"
            placeholder="Type your command..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            autoFocus
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="group p-3 bg-black text-white rounded-full hover:bg-gray-800 disabled:bg-gray-200 transition-all active:scale-95"
          >
            <ArrowUp
              size={20}
              className="group-hover:-translate-y-0.5 transition-transform"
            />
          </button>
        </div>
      </div>
    </div>
  );
}

// --- Sub-Component: Minimalist Structured Response ---
const StructuredResponse = ({ data }) => {
  if (data.status === "error") {
    return (
      <div className="border border-black p-4 font-mono text-sm">
        ERROR: {data.message}
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full animate-in fade-in slide-in-from-bottom-2 duration-500">
      {data.task_title && (
        <h2 className="text-2xl font-light tracking-tight border-l-4 border-black pl-4 py-1">
          {data.task_title}
        </h2>
      )}

      {/* Steps List */}
      <div className="space-y-8 pl-1">
        {data.steps?.map((step, idx) => (
          <div
            key={idx}
            className="group relative pl-8 border-l border-gray-200 hover:border-black transition-colors duration-300"
          >
            {/* Step Number */}
            <span className="absolute -left-[9px] top-0 w-[17px] h-[17px] bg-white border border-gray-300 group-hover:border-black group-hover:bg-black rounded-full transition-colors duration-300"></span>

            <div className="flex flex-col gap-3">
              <div className="flex items-baseline justify-between">
                <p className="text-base font-normal leading-relaxed text-gray-900">
                  <span className="font-mono text-xs text-gray-400 mr-2 uppercase tracking-wide">
                    Step {step.step}
                  </span>
                  {step.instruction}
                </p>
              </div>

              {/* Citations */}
              {step.chunks && step.chunks.length > 0 && (
                <div className="font-mono text-[10px] text-gray-300">
                  REF: {step.chunks.map((c) => `[#${c}]`).join(" ")}
                </div>
              )}

              {/* Image Rendering: Supports Array (Top 3) or Single String */}
              {step.images && (
                <div className="mt-3 flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-300">
                  {/* Handle if backend sends List vs String */}
                  {Array.isArray(step.images)
                    ? step.images.map((imgUrl, imgIdx) => (
                        <div
                          key={imgIdx}
                          className="shrink-0 border border-gray-200 p-1 bg-gray-50 rounded-sm"
                        >
                          <img
                            src={convertPathToUrl(imgUrl)}
                            alt={`Step visual ${imgIdx + 1}`}
                            className="h-32 w-auto object-contain transition-transform hover:scale-105"
                            onError={(e) => {
                              e.target.style.display = "none";
                            }}
                          />
                        </div>
                      ))
                    : // Fallback for single string
                      step.images !== "null" && (
                        <div className="shrink-0 border border-gray-200 p-1 bg-gray-50 rounded-sm">
                          <img
                            src={convertPathToUrl(step.images)}
                            alt="Visual Aid"
                            className="h-32 w-auto object-contain transition-transform hover:scale-105"
                            onError={(e) => {
                              e.target.style.display = "none";
                            }}
                          />
                        </div>
                      )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// --- HELPER: FIXED IMAGE PATH ---
function convertPathToUrl(localPath) {
  if (!localPath || localPath === "null") return "";
  if (typeof localPath !== "string") return "";

  try {
    const filename = localPath.split(/[\\/]/).pop();
    // Use 'final_cleaned_dataset' route
    return `${API_URL}/final_cleaned_dataset/${filename}`;
  } catch (err) {
    console.error("Error parsing image path:", err);
    return "";
  }
}

export default App;
