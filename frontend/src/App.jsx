import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, Bot, User, Loader2, ArrowUp, CheckCircle, AlertCircle, ShieldCheck } from "lucide-react";

const API_URL = "http://localhost:5000";
const AVAILABLE_MODELS = [
    { id: "WA5471ABP", name: "WA5471ABP (Top Load Washer)" },
    { id: "WF5M5100AW", name: "WF5M5100AW (Front Load Washer)" },
    { id: "WF350ANR", name: "WF350ANR (Front Load Washer)" },
    { id: "DC68", name: "DC68 Series Washer" },
    { id: "General", name: "General / Auto-Detect" }
];

function App() {
    const [query, setQuery] = useState("");
    const [selectedModel, setSelectedModel] = useState("WA5471ABP");
    const [messages, setMessages] = useState([
        {
            type: "bot",
            content: {
                status: "success",
                task_title: "Samsung Multimodal Repair Assistant",
                grounding_confidence: 10,
                model: "WA5471ABP",
                steps: [
                    {
                        step: 0,
                        instruction:
                            "System initialized with grounded retrieval & verification. Select a model or type your repair/maintenance query below.",
                        match_type: "system_ready"
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

    const handleSend = async (overrideQuery = null, overrideModel = null) => {
        const textToSend = overrideQuery || query;
        const modelToSend = overrideModel || selectedModel;
        if (!textToSend.trim()) return;

        // Add User Message
        const newMsgs = [...messages, { type: "user", content: textToSend, model: modelToSend }];
        setMessages(newMsgs);
        if (!overrideQuery) setQuery("");
        setLoading(true);

        try {
            const res = await axios.post(`${API_URL}/api/chat`, {
                query: textToSend,
                model: modelToSend,
                allow_generic: modelToSend === "General"
            });
            setMessages((prev) => [
                ...prev,
                { type: "bot", content: res.data },
            ]);
        } catch (error) {
            const errorMessage =
                error.response?.data?.message ||
                error.response?.data?.error ||
                "Connection failure. Ensure backend server is running.";
            setMessages((prev) => [
                ...prev,
                {
                    type: "bot",
                    content: {
                        status: "error",
                        message: errorMessage,
                    },
                },
            ]);
        }
        setLoading(false);
    };

    const handleSelectDisambiguation = (modelId) => {
        setSelectedModel(modelId);
        // Find last user query
        const lastUserMsg = [...messages].reverse().find(m => m.type === "user");
        if (lastUserMsg && typeof lastUserMsg.content === "string") {
            handleSend(lastUserMsg.content, modelId);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-600 selection:text-white">
            {/* Header */}
            <header className="border-b border-slate-200 bg-white/90 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 bg-blue-600 text-white flex items-center justify-center rounded-lg shadow-sm">
                        <Bot size={20} />
                    </div>
                    <div>
                        <h1 className="text-base font-semibold tracking-tight text-slate-900">
                            SAMSUNG PRISM REPAIR RAG
                        </h1>
                        <p className="text-xs text-slate-500 font-mono">Grounded Multimodal Guidance</p>
                    </div>
                </div>

                {/* Model Selector Slot */}
                <div className="flex items-center gap-2">
                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 font-mono">
                        Target Model:
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="text-xs font-medium bg-slate-100 border border-slate-300 rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                    >
                        {AVAILABLE_MODELS.map((m) => (
                            <option key={m.id} value={m.id}>
                                {m.name}
                            </option>
                        ))}
                    </select>
                </div>
            </header>

            {/* Chat Container */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 max-w-4xl mx-auto w-full">
                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`flex flex-col ${
                            msg.type === "user" ? "items-end" : "items-start"
                        }`}
                    >
                        {/* Message Content */}
                        <div
                            className={`max-w-[95%] sm:max-w-[88%] rounded-xl p-0 ${
                                msg.type === "user"
                                    ? "text-right"
                                    : "text-left w-full"
                            }`}
                        >
                            {typeof msg.content === "string" ? (
                                <div className="inline-block bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm text-sm font-medium shadow-sm">
                                    {msg.content}
                                    {msg.model && (
                                        <div className="text-[10px] text-blue-200 mt-1 font-mono">
                                            Model: {msg.model}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <StructuredResponse
                                    data={msg.content}
                                    onSelectModel={handleSelectDisambiguation}
                                />
                            )}
                        </div>

                        {/* Timestamp / Role Label */}
                        <span className="text-[10px] uppercase tracking-widest text-slate-400 mt-1.5 font-mono px-1">
                            {msg.type === "user" ? "User Query" : "Grounded Response"}
                        </span>
                    </div>
                ))}

                {loading && (
                    <div className="flex items-center gap-3 text-xs font-mono text-slate-500 bg-white border border-slate-200 rounded-lg px-4 py-3 shadow-sm w-fit animate-pulse">
                        <Loader2 size={16} className="animate-spin text-blue-600" />
                        RETRIEVING & GROUNDING STEPS...
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 sm:p-5 bg-white border-t border-slate-200 shadow-lg">
                <div className="max-w-4xl mx-auto flex items-center gap-3">
                    <input
                        type="text"
                        className="flex-1 bg-slate-50 border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all placeholder:text-slate-400"
                        placeholder={`Ask a repair/maintenance question (e.g. "How to clean the debris filter?") for ${selectedModel}...`}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                        autoFocus
                    />
                    <button
                        onClick={() => handleSend()}
                        disabled={loading || !query.trim()}
                        className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 transition-all shadow-sm active:scale-95 flex items-center justify-center"
                    >
                        <ArrowUp size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
}

// --- Sub-Component: Minimalist Structured Response ---
const StructuredResponse = ({ data, onSelectModel }) => {
    if (data.status === "error") {
        return (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-xs font-mono flex items-start gap-2.5">
                <AlertCircle size={16} className="shrink-0 mt-0.5 text-red-500" />
                <div>
                    <span className="font-bold">ERROR: </span>
                    {data.message}
                </div>
            </div>
        );
    }

    if (data.status === "disambiguation_required") {
        return (
            <div className="bg-amber-50 border border-amber-200 text-amber-900 rounded-xl p-5 text-sm space-y-3">
                <div className="flex items-center gap-2 font-semibold text-amber-800">
                    <AlertCircle size={18} className="text-amber-600" />
                    Model Disambiguation Needed
                </div>
                <p className="text-xs text-amber-700 leading-relaxed">
                    {data.message}
                </p>
                <div className="flex flex-wrap gap-2 pt-1">
                    {data.available_models?.map((modelId) => (
                        <button
                            key={modelId}
                            onClick={() => onSelectModel(modelId)}
                            className="text-xs font-mono font-medium bg-white hover:bg-amber-100 border border-amber-300 text-amber-900 px-3 py-1.5 rounded-lg shadow-sm transition-colors"
                        >
                            {modelId}
                        </button>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-sm space-y-6 animate-in fade-in duration-300">
            {/* Task Header & Metrics Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-4">
                <div>
                    <span className="text-[10px] uppercase font-mono font-bold tracking-wider text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                        Model: {data.model || "General"}
                    </span>
                    <h2 className="text-xl font-bold tracking-tight text-slate-900 mt-1">
                        {data.task_title || "Procedure Guide"}
                    </h2>
                </div>

                {/* Grounding & Confidence Badges */}
                <div className="flex items-center gap-2">
                    {data.grounding_confidence && (
                        <div className="flex items-center gap-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-1 rounded-full text-xs font-mono font-semibold">
                            <ShieldCheck size={14} className="text-emerald-600" />
                            Grounding: {data.grounding_confidence}/10
                        </div>
                    )}
                    {data.repetition_rate !== undefined && (
                        <div className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-1 rounded-md">
                            Repetition: {data.repetition_rate}%
                        </div>
                    )}
                </div>
            </div>

            {/* Steps Sequence */}
            <div className="space-y-6">
                {data.steps?.map((step, idx) => (
                    <div
                        key={idx}
                        className="group relative pl-7 border-l-2 border-slate-200 hover:border-blue-500 transition-colors duration-200"
                    >
                        {/* Step Marker */}
                        <span className="absolute -left-[9px] top-0.5 w-[16px] h-[16px] bg-white border-2 border-slate-300 group-hover:border-blue-600 group-hover:bg-blue-600 rounded-full transition-colors duration-200"></span>

                        <div className="space-y-2.5">
                            {/* Step Text */}
                            <div className="flex items-baseline justify-between">
                                <p className="text-sm font-normal leading-relaxed text-slate-800">
                                    <span className="font-mono text-xs font-bold text-slate-500 mr-2 uppercase tracking-wide">
                                        Step {step.step}:
                                    </span>
                                    {step.instruction}
                                </p>
                            </div>

                            {/* Badges: Match Type & Citations */}
                            <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
                                {step.chunk_ids && step.chunk_ids.length > 0 && (
                                    <span className="text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                                        REF: {step.chunk_ids.map((c) => `[Chunk ${c}]`).join(" ")}
                                    </span>
                                )}
                                {step.match_type === "direct_page_link" && (
                                    <span className="text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded flex items-center gap-1 font-semibold">
                                        <CheckCircle size={10} /> Tier 1: Direct Manual Match ({step.image_confidence})
                                    </span>
                                )}
                                {step.match_type === "semantic_fallback" && (
                                    <span className="text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded font-semibold">
                                        Tier 2: Semantic Match ({step.image_confidence})
                                    </span>
                                )}
                                {step.match_type === "rejected_low_confidence" && (
                                    <span className="text-slate-400 bg-slate-50 border border-slate-200 px-2 py-0.5 rounded italic">
                                        Text Only (No diagram needed)
                                    </span>
                                )}
                            </div>

                            {/* Image Visual Aid */}
                            {step.images && Array.isArray(step.images) && step.images.length > 0 && (
                                <div className="pt-1 flex gap-3 overflow-x-auto pb-1">
                                    {step.images.map((imgUrl, imgIdx) => (
                                        <div
                                            key={imgIdx}
                                            className="shrink-0 border border-slate-200 p-1.5 bg-slate-50 rounded-lg shadow-sm hover:shadow transition-shadow"
                                        >
                                            <img
                                                src={convertPathToUrl(imgUrl)}
                                                alt={`Visual instruction for Step ${step.step}`}
                                                className="h-36 w-auto object-contain rounded transition-transform hover:scale-105"
                                                onError={(e) => {
                                                    e.target.style.display = "none";
                                                }}
                                            />
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// --- HELPER: CONVERT PATH TO URL ---
function convertPathToUrl(localPath) {
    if (!localPath || localPath === "null") return "";
    if (typeof localPath !== "string") return "";

    try {
        const filename = localPath.split(/[\\/]/).pop();
        return `${API_URL}/extracted_images/${filename}`;
    } catch (err) {
        console.error("Error parsing image path:", err);
        return "";
    }
}

export default App;
