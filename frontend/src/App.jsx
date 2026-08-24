import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
    Bot,
    User,
    Loader2,
    ArrowUp,
    CheckCircle2,
    AlertCircle,
    AlertTriangle,
    ShieldCheck,
    Radio,
    BookOpen,
    Info,
    Check
} from "lucide-react";

const API_URL = "http://localhost:5000";
const DEFAULT_MODELS = [
    { id: "WA5471ABP", name: "WA5471ABP (Top Load Washer)" },
    { id: "General", name: "General / Auto-Detect" }
];

function App() {
    const [query, setQuery] = useState("");
    const [selectedModel, setSelectedModel] = useState("WA5471ABP");
    const [backendStatus, setBackendStatus] = useState({ online: false, checking: true, info: null });
    const [availableModels, setAvailableModels] = useState(DEFAULT_MODELS);
    const [messages, setMessages] = useState([
        {
            type: "bot",
            content: {
                status: "success",
                task_title: "Samsung Grounded Repair Assistant",
                grounding: {
                    confidence: "high",
                    grounded: true
                },
                model: "WA5471ABP",
                steps: [
                    {
                        step_number: 1,
                        instruction:
                            "System initialized with grounded Qdrant retrieval, Groq inference & visual guidance. Select your Samsung model above or type a troubleshooting query below.",
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

    // Check backend connection on mount and fetch models
    useEffect(() => {
        const checkBackendHealth = async () => {
            try {
                const [healthRes, modelsRes] = await Promise.allSettled([
                    axios.get(`${API_URL}/api/health`, { timeout: 4000 }),
                    axios.get(`${API_URL}/api/models`, { timeout: 4000 })
                ]);

                if (healthRes.status === "fulfilled" && healthRes.value.data?.status === "healthy") {
                    setBackendStatus({
                        online: true,
                        checking: false,
                        info: healthRes.value.data
                    });
                } else {
                    setBackendStatus({ online: false, checking: false, info: null });
                }

                if (modelsRes.status === "fulfilled" && modelsRes.value.data?.models) {
                    const serverModels = modelsRes.value.data.models.map(m => {
                        if (typeof m === "string") return { id: m, name: `${m} Washer` };
                        return { id: m.display_name || m.canonical_model, name: `${m.display_name || m.canonical_model} Washer` };
                    });
                    if (!serverModels.some(m => m.id === "General")) {
                        serverModels.push({ id: "General", name: "General / Auto-Detect" });
                    }
                    setAvailableModels(serverModels);
                }
            } catch {
                setBackendStatus({ online: false, checking: false, info: null });
            }
        };

        checkBackendHealth();
        const interval = setInterval(checkBackendHealth, 30000);
        return () => clearInterval(interval);
    }, []);

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
                model: modelToSend === "General" ? "" : modelToSend
            });
            setMessages((prev) => [
                ...prev,
                { type: "bot", content: res.data },
            ]);
            setBackendStatus(prev => ({ ...prev, online: true }));
        } catch (error) {
            const errorMessage =
                error.response?.data?.message ||
                error.response?.data?.error ||
                "Connection failure. Ensure backend server is running on http://localhost:5000.";
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

    const handleSelectDisambiguation = (modelItem) => {
        const modelId = typeof modelItem === "object" ? (modelItem.display_name || modelItem.canonical_model) : modelItem;
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
            <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md px-6 py-3.5 flex items-center justify-between sticky top-0 z-10 shadow-sm">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 bg-blue-600 text-white flex items-center justify-center rounded-xl shadow-sm">
                        <Bot size={20} />
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h1 className="text-sm font-bold tracking-tight text-slate-900">
                                SAMSUNG PRISM REPAIR RAG
                            </h1>
                            {/* Backend Live Connection Badge */}
                            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono border transition-all">
                                {backendStatus.checking ? (
                                    <span className="flex items-center gap-1 text-slate-500 bg-slate-100 border-slate-200 px-1.5 py-0.5 rounded-full">
                                        <Loader2 size={10} className="animate-spin" /> Checking Backend
                                    </span>
                                ) : backendStatus.online ? (
                                    <span className="flex items-center gap-1 text-emerald-700 bg-emerald-50 border-emerald-200 px-1.5 py-0.5 rounded-full font-medium">
                                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                                        Connected ({backendStatus.info?.llm_provider || "Groq"} + {backendStatus.info?.retrieval || "Qdrant"})
                                    </span>
                                ) : (
                                    <span className="flex items-center gap-1 text-rose-700 bg-rose-50 border-rose-200 px-1.5 py-0.5 rounded-full font-medium">
                                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                                        Backend Disconnected (localhost:5000)
                                    </span>
                                )}
                            </div>
                        </div>
                        <p className="text-xs text-slate-500 font-mono">Grounded Multimodal Guidance & Stage 8 Orchestration</p>
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
                        className="text-xs font-medium bg-slate-100 border border-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer shadow-sm"
                    >
                        {availableModels.map((m) => (
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
                            className={`max-w-[95%] sm:max-w-[90%] rounded-xl p-0 ${
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
                            {msg.type === "user" ? "User Query" : "Grounded Technical Response"}
                        </span>
                    </div>
                ))}

                {loading && (
                    <div className="flex items-center gap-3 text-xs font-mono text-slate-600 bg-white border border-slate-200 rounded-xl px-4 py-3 shadow-sm w-fit animate-pulse">
                        <Loader2 size={16} className="animate-spin text-blue-600" />
                        RETRIEVING FROM QDRANT & GROUNDING EVIDENCE WITH GROQ...
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 sm:p-5 bg-white border-t border-slate-200 shadow-lg">
                <div className="max-w-4xl mx-auto flex items-center gap-3">
                    <input
                        type="text"
                        className="flex-1 bg-slate-50 border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all placeholder:text-slate-400 shadow-inner"
                        placeholder={`Ask a repair/maintenance question (e.g. "How to clean the water hose filter?") for ${selectedModel}...`}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                        autoFocus
                    />
                    <button
                        onClick={() => handleSend()}
                        disabled={loading || !query.trim()}
                        className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 transition-all shadow-sm active:scale-95 flex items-center justify-center cursor-pointer disabled:cursor-not-allowed"
                        title="Send Query"
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
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-xs font-mono flex items-start gap-2.5 shadow-sm">
                <AlertCircle size={16} className="shrink-0 mt-0.5 text-red-500" />
                <div>
                    <span className="font-bold">ERROR: </span>
                    {data.message || "An unexpected error occurred."}
                </div>
            </div>
        );
    }

    if (data.status === "disambiguation_required") {
        return (
            <div className="bg-amber-50 border border-amber-200 text-amber-900 rounded-xl p-5 text-sm space-y-3 shadow-sm">
                <div className="flex items-center gap-2 font-semibold text-amber-800">
                    <AlertCircle size={18} className="text-amber-600" />
                    Model Disambiguation Needed
                </div>
                <p className="text-xs text-amber-700 leading-relaxed">
                    {data.message}
                </p>
                <div className="flex flex-wrap gap-2 pt-1">
                    {data.available_models?.map((modelItem, idx) => {
                        const mId = typeof modelItem === "object" ? (modelItem.display_name || modelItem.canonical_model) : modelItem;
                        return (
                            <button
                                key={idx}
                                onClick={() => onSelectModel(modelItem)}
                                className="text-xs font-mono font-medium bg-white hover:bg-amber-100 border border-amber-300 text-amber-900 px-3 py-1.5 rounded-lg shadow-sm transition-colors cursor-pointer"
                            >
                                {mId}
                            </button>
                        );
                    })}
                </div>
            </div>
        );
    }

    if (data.status === "model_conflict") {
        return (
            <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded-xl p-5 text-sm space-y-3 shadow-sm">
                <div className="flex items-center gap-2 font-semibold text-rose-800">
                    <AlertTriangle size={18} className="text-rose-600" />
                    Model Conflict Detected
                </div>
                <p className="text-xs text-rose-700 leading-relaxed">
                    {data.message}
                </p>
                {data.models_detected && data.models_detected.length > 0 && (
                    <div className="flex flex-wrap gap-2 pt-1">
                        <span className="text-xs font-mono text-rose-600">Detected:</span>
                        {data.models_detected.map((m, idx) => (
                            <span key={idx} className="text-xs font-mono font-semibold bg-white border border-rose-300 px-2 py-0.5 rounded text-rose-800">
                                {m}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        );
    }

    if (data.status === "no_results") {
        return (
            <div className="bg-slate-50 border border-slate-200 text-slate-700 rounded-xl p-5 text-xs font-mono space-y-3">
                <div className="flex items-center gap-2 font-semibold text-slate-800 text-sm">
                    <Info size={16} className="text-slate-500" />
                    No Grounded Documentation Found
                </div>
                {data.warning && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-3 text-xs flex items-start gap-2">
                        <AlertTriangle size={14} className="shrink-0 text-amber-600 mt-0.5" />
                        <div>
                            <span className="font-bold font-mono text-[10px] uppercase block">Notice</span>
                            {data.warning.message || data.warning}
                        </div>
                    </div>
                )}
                <p className="text-slate-600 leading-relaxed">
                    {data.message || "No technical manual evidence was found for this query and model."}
                </p>
            </div>
        );
    }

    // Extract grounding confidence
    const confidence =
        data.grounding?.confidence ||
        (data.grounding_confidence ? `${data.grounding_confidence}/10` : null);

    const isHighConfidence = typeof confidence === "string" && (confidence.toLowerCase() === "high" || confidence.includes("10") || confidence.includes("9") || confidence.includes("8"));
    const isMediumConfidence = typeof confidence === "string" && (confidence.toLowerCase() === "medium" || confidence.includes("7") || confidence.includes("6") || confidence.includes("5"));

    const isGeneric = data.guidance_scope === "generic" || data.model_known === false;

    return (
        <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-sm space-y-6 animate-in fade-in duration-300">
            {/* Task Header & Metrics Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-4">
                <div>
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase font-mono font-bold tracking-wider text-blue-600 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded">
                            Model: {data.model || "General"}
                        </span>
                        {isGeneric ? (
                            <span className="text-[10px] uppercase font-mono font-semibold tracking-wider text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded flex items-center gap-1">
                                <AlertTriangle size={10} className="text-amber-600" /> Generic Samsung Guidance
                            </span>
                        ) : (
                            <span className="text-[10px] uppercase font-mono font-semibold tracking-wider text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded flex items-center gap-1">
                                <CheckCircle2 size={10} className="text-emerald-600" /> Model-Specific Verified
                            </span>
                        )}
                    </div>
                    <h2 className="text-lg sm:text-xl font-bold tracking-tight text-slate-900 mt-1.5">
                        {data.task_title || "Procedure Guide"}
                    </h2>
                </div>

                {/* Grounding & Confidence Badges */}
                <div className="flex items-center gap-2">
                    {confidence && (
                        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-semibold border ${
                            isHighConfidence
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                : isMediumConfidence
                                ? "bg-amber-50 text-amber-700 border-amber-200"
                                : "bg-slate-100 text-slate-700 border-slate-200"
                        }`}>
                            <ShieldCheck size={14} className={isHighConfidence ? "text-emerald-600" : isMediumConfidence ? "text-amber-600" : "text-slate-500"} />
                            Grounding: {typeof confidence === "string" ? confidence.toUpperCase() : confidence}
                        </div>
                    )}
                    {data.grounding?.source_problem_ids && data.grounding.source_problem_ids.length > 0 && (
                        <span className="hidden sm:inline text-[10px] font-mono text-slate-500 bg-slate-50 border border-slate-200 px-2 py-1 rounded-md">
                            Source: {data.grounding.source_problem_ids.length} problem(s)
                        </span>
                    )}
                </div>
            </div>

            {/* Authoritative Warning Callout (Displayed prominently before instructions) */}
            {data.warning && (
                <div className="bg-amber-50/90 border-2 border-amber-300 text-amber-950 rounded-xl p-4 text-xs flex items-start gap-3 shadow-sm">
                    <AlertTriangle size={20} className="shrink-0 text-amber-600 mt-0.5" />
                    <div>
                        <span className="font-bold text-xs uppercase font-mono text-amber-900 block">
                            ⚠️ Unknown Model Notice
                        </span>
                        <p className="text-xs text-amber-800 mt-1 leading-relaxed font-medium">
                            {data.warning.message || data.warning}
                        </p>
                    </div>
                </div>
            )}

            {/* Limitations Callout (if any) */}
            {data.limitations && data.limitations.length > 0 && (
                <div className="bg-amber-50/70 border border-amber-200 text-amber-900 rounded-lg p-3 text-xs flex items-start gap-2">
                    <AlertTriangle size={15} className="shrink-0 text-amber-600 mt-0.5" />
                    <div>
                        <span className="font-semibold block font-mono text-[10px] uppercase text-amber-800">Documentation Note</span>
                        {data.limitations.map((lim, lIdx) => (
                            <p key={lIdx} className="text-amber-800 mt-0.5">{lim}</p>
                        ))}
                    </div>
                </div>
            )}

            {/* Steps Sequence */}
            <div className="space-y-6">
                {data.steps && data.steps.length > 0 ? (
                    data.steps.map((step, idx) => {
                        const stepNum = step.step_number ?? step.step ?? (idx + 1);
                        const chunkIds = step.source?.chunk_ids || step.chunk_ids || [];
                        const pages = step.source?.pages || [];

                        return (
                            <div
                                key={step.step_id || idx}
                                className="group relative pl-7 border-l-2 border-slate-200 hover:border-blue-500 transition-colors duration-200"
                            >
                                {/* Step Marker */}
                                <span className="absolute -left-[9px] top-0.5 w-[16px] h-[16px] bg-white border-2 border-slate-300 group-hover:border-blue-600 group-hover:bg-blue-600 rounded-full transition-colors duration-200"></span>

                                <div className="space-y-2.5">
                                    {/* Step Text */}
                                    <div className="flex items-baseline justify-between">
                                        <p className="text-sm font-normal leading-relaxed text-slate-800">
                                            <span className="font-mono text-xs font-bold text-slate-600 mr-2 uppercase tracking-wide">
                                                Step {stepNum}:
                                            </span>
                                            {step.instruction || step.step_text}
                                        </p>
                                    </div>

                                    {/* Safety Warning (if present) */}
                                    {step.safety_warning && (
                                        <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-lg p-2.5 text-xs flex items-start gap-2">
                                            <AlertTriangle size={14} className="shrink-0 text-rose-600 mt-0.5" />
                                            <div>
                                                <span className="font-bold text-[10px] uppercase font-mono text-rose-700 block">Safety Warning:</span>
                                                {step.safety_warning}
                                            </div>
                                        </div>
                                    )}

                                    {/* Badges: Citations & Manual Provenance */}
                                    <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
                                        {chunkIds.length > 0 && (
                                            <span className="text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded">
                                                REF: {chunkIds.map((c) => `[Chunk ${c.split('_').pop()}]`).join(" ")}
                                            </span>
                                        )}
                                        {pages.length > 0 && (
                                            <span className="text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded flex items-center gap-1">
                                                <BookOpen size={10} /> Page {pages.join(", ")}
                                            </span>
                                        )}
                                        {step.match_type === "direct_page_link" && (
                                            <span className="text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded flex items-center gap-1 font-semibold">
                                                <CheckCircle2 size={10} /> Tier 1: Direct Match
                                            </span>
                                        )}
                                    </div>

                                    {/* Image Visual Aid */}
                                    {step.images && Array.isArray(step.images) && step.images.length > 0 && (
                                        <div className="pt-1.5 flex gap-3 overflow-x-auto pb-1">
                                            {step.images.map((img, imgIdx) => {
                                                const imgUrl = convertPathToUrl(img);
                                                if (!imgUrl) return null;

                                                const isStepMatch = typeof img === "object" && img.step_match;
                                                const isImgGeneric = typeof img === "object" && img.image_scope === "generic";

                                                return (
                                                    <div
                                                        key={imgIdx}
                                                        className="shrink-0 border border-slate-200 p-2 bg-slate-50 rounded-xl shadow-sm hover:shadow transition-shadow max-w-sm"
                                                    >
                                                        <img
                                                            src={imgUrl}
                                                            alt={`Visual instruction for Step ${stepNum}`}
                                                            className="h-36 w-auto object-contain rounded-lg transition-transform hover:scale-105 bg-white"
                                                            onError={(e) => {
                                                                e.target.style.display = "none";
                                                            }}
                                                        />
                                                        {isStepMatch && !isImgGeneric && (
                                                            <div className="mt-1.5 text-[9px] font-mono text-emerald-700 flex items-center gap-1 font-medium">
                                                                <CheckCircle2 size={10} /> Step Verified Diagram
                                                            </div>
                                                        )}
                                                        {isImgGeneric && (
                                                            <div className="mt-1.5 text-[9px] font-mono text-slate-500 flex items-center gap-1">
                                                                <Info size={10} /> Generic Visual Aid
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })
                ) : (
                    <div className="text-xs font-mono text-slate-500 italic py-2">
                        No specific procedure steps available in the retrieval evidence for this symptom.
                    </div>
                )}
            </div>
        </div>
    );
};

// --- HELPER: CONVERT PATH TO URL ---
function convertPathToUrl(img) {
    if (!img) return "";
    
    // Case 1: Object returned by backend
    if (typeof img === "object") {
        if (img.url && typeof img.url === "string") {
            if (img.url.startsWith("http://") || img.url.startsWith("https://")) {
                return img.url;
            }
            const cleanUrl = img.url.startsWith("/") ? img.url : `/${img.url}`;
            return `${API_URL}${cleanUrl}`;
        }
        if (img.file_path && typeof img.file_path === "string") {
            const filename = img.file_path.split(/[\\/]/).pop();
            return `${API_URL}/generated_step_images_20260824_0052/${filename}`;
        }
        return "";
    }

    // Case 2: Direct string URL / path
    if (typeof img === "string") {
        if (img.startsWith("http://") || img.startsWith("https://")) return img;
        if (img.startsWith("/")) return `${API_URL}${img}`;
        const filename = img.split(/[\\/]/).pop();
        return `${API_URL}/generated_step_images_20260824_0052/${filename}`;
    }

    return "";
}

export default App;
