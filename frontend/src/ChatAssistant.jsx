import { useState, useEffect, useRef } from "react"
import {
  Bot,
  Send,
  X,
  FileText,
  AlertTriangle,
  RefreshCw,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Languages,
  RotateCcw,
  Sprout,
  Check,
  Edit3,
  Brain,
  GitCompare,
  Award,
  CheckCircle2,
} from "lucide-react"

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

let messageSequence = 0
function generateId(prefix = "msg") {
  messageSequence += 1
  return `${prefix}-${messageSequence}-${Date.now().toString(36)}`
}

function getInitialPrediction(context) {
  if (!context?.predictions || context.predictions.length === 0) return null
  const topPred = context.predictions[0]
  const rawLabel = topPred.label
  const formatted = rawLabel.replace(/___/g, " - ").replace(/_/g, " ")
  const confPercent = Math.round(topPred.confidence * 100)

  return {
    rawLabel,
    formatted,
    confidence: topPred.confidence,
    percentage: `${confPercent}%`,
    allPredictions: context.predictions,
    detectedDiseases: context.detectedDiseases || [rawLabel],
  }
}

function getInitialCrop(prediction) {
  if (!prediction) return ""
  const raw = prediction.rawLabel
  if (raw.includes("___")) {
    return raw.split("___")[0].replace(/_/g, " ").replace(/\s*\([^)]*\)/g, "").trim()
  }
  return ""
}

function getInitialWelcomeMessage(prediction, language) {
  const isHi = language === "hi"
  if (prediction) {
    const isLowConfidence = prediction.confidence < 0.50
    const text = isHi
      ? isLowConfidence
        ? `नमस्ते! आपके पत्ते के स्कैन में **${prediction.formatted}** (${prediction.percentage}) की संभावना दिखी है।\n\n⚠️ **महत्वपूर्ण सूचना:** यह केवल एक प्रारंभिक दृश्य संकेत है, कोई निश्चित प्रयोगशाला निदान नहीं। सटीक पहचान के लिए आइए कुछ बुनियादी लक्षण साझा करें:`
        : `नमस्ते! मैं आपका **LeafSense स्मार्ट फार्मिंग सहायक** हूँ।\n\nआपके पत्ते के स्कैन में **${prediction.formatted}** (${prediction.percentage} संभावना) पाया गया है। कृपया ध्यान दें कि यह प्रारंभिक दृश्य संकेत है, प्रयोगशाला निदान नहीं। आप इसके लक्षण, रोकथाम या जैविक उपचार के बारे में क्या जानना चाहते हैं?`
      : isLowConfidence
        ? `Hello! I noticed your leaf diagnosis shows preliminary similarity to **${prediction.formatted}** (${prediction.percentage} confidence).\n\n⚠️ **Important Note:** This prediction has lower certainty and is only a preliminary visual indicator, not a confirmed laboratory diagnosis. Let's verify a few practical observations together:`
        : `Hello! I am your **LeafSense Smart Farming Assistant**, powered by **IBM Granite 4**.\n\nYour preliminary leaf scan indicates **${prediction.formatted}** (${prediction.percentage} confidence). Please remember this visual scan is a preliminary indicator, not a confirmed laboratory diagnosis. What would you like to verify or treat first?`

    return [
      {
        id: generateId("welcome"),
        sender: "agent",
        text,
        time: "Just now",
        quickReplies: isHi
          ? ["निचली पत्तियों पर गोल छल्ले", "पत्तियों के नीचे सफेद फफूंद", "छोड़ें / सीधा उपचार बताएं"]
          : ["Target rings on lower leaves", "Water-soaked patches", "Skip / Show Guidance"],
      },
    ]
  }

  const text = isHi
    ? "नमस्ते! मैं आपका **LeafSense स्मार्ट फार्मिंग सहायक** हूँ, जो **IBM Granite 4** द्वारा संचालित है।\n\nमैं आपकी फसल के रोगों, लक्षणों की पहचान, जैविक रोकथाम और कृषि ज्ञानकोश से सत्यापित सलाह में मदद कर सकता हूँ। आप किस फसल या समस्या के बारे में पूछना चाहते हैं?"
    : "Hello! I am your **LeafSense Smart Farming Assistant**, powered by **IBM Granite 4**.\n\nI can help you analyze crop symptoms, verify disease patterns, explore organic prevention, and consult our verified Agricultural Knowledge Base. Which crop or field issue would you like to discuss today?"
  return [
    {
      id: generateId("welcome"),
      sender: "agent",
      text,
      time: "Just now",
      quickReplies: isHi
        ? ["टमाटर में झुलसा रोग की जांच", "गेहूं में पीला रतुआ", "जैविक कीटनाशक की तैयारी", "फसल चक्र के लाभ"]
        : ["Check tomato blight symptoms", "Yellow rust in wheat", "Organic pest control methods", "Crop rotation benefits"],
    },
  ]
}

const STAGE_PROGRESS = {
  crop_context: { step: 1, percent: 25, label_en: "1/4 Understanding Your Crop", label_hi: "1/4 फसल की जानकारी" },
  symptom_progression: { step: 2, percent: 50, label_en: "2/4 Analyzing Symptoms & Spread", label_hi: "2/4 लक्षण व फैलाव का विश्लेषण" },
  field_environment: { step: 3, percent: 75, label_en: "3/4 Field & Weather Conditions", label_hi: "3/4 खेत की स्थिति व मौसम" },
  advisory_management: { step: 4, percent: 100, label_en: "4/4 Personalized Advisory & Management", label_hi: "4/4 सटीक परामर्श व समाधान" },
}

export default function ChatAssistant({
  isOpen,
  onClose,
  initialContext = null,
  uiLanguage = "en",
  darkMode = false,
}) {
  const [chatLanguage, setChatLanguage] = useState(() => (uiLanguage === "hi" ? "hi" : "en"))
  const [attachedPrediction, setAttachedPrediction] = useState(() => getInitialPrediction(initialContext))
  const [crop, setCrop] = useState(() => getInitialCrop(attachedPrediction))
  const [location, setLocation] = useState("")

  // Multi-turn conversation state
  const [conversationId, setConversationId] = useState(() => `leafsense-${Math.random().toString(36).substring(2, 10)}`)
  const [consultationStage, setConsultationStage] = useState("crop_context")
  const [stageLabel, setStageLabel] = useState("")
  const [farmerProfile, setFarmerProfile] = useState(() => ({
    crop: getInitialCrop(attachedPrediction) || null,
    location: null,
    variety: null,
    plant_age: null,
    growth_stage: null,
    symptoms: null,
    first_noticed: null,
    is_spreading: null,
    affected_percentage: null,
    irrigation_type: null,
    recent_weather: null,
    treatments_applied: null,
  }))

  const [showProfileSummary, setShowProfileSummary] = useState(false)
  const [isEditingProfile, setIsEditingProfile] = useState(false)
  const [editCrop, setEditCrop] = useState(() => crop)
  const [editLocation, setEditLocation] = useState(() => location)

  const [activeQuickReplies, setActiveQuickReplies] = useState(() =>
    attachedPrediction
      ? (uiLanguage === "hi"
          ? ["निचली पत्तियों पर गोल छल्ले", "पत्तियों के नीचे सफेद फफूंद", "छोड़ें / सीधा उपचार बताएं"]
          : ["Target rings on lower leaves", "Water-soaked patches", "Skip / Show Guidance"])
      : (uiLanguage === "hi"
          ? ["टमाटर में झुलसा रोग", "गेहूं में पीला रतुआ", "जैविक रोकथाम"]
          : ["Check tomato blight", "Yellow rust in wheat", "Organic remedies"])
  )

  const [expandedSources, setExpandedSources] = useState({})
  const [expandedReasoning, setExpandedReasoning] = useState({})
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastFailedQuery, setLastFailedQuery] = useState("")

  const [messages, setMessages] = useState(() =>
    getInitialWelcomeMessage(attachedPrediction, uiLanguage === "hi" ? "hi" : "en")
  )

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, loading, isOpen, activeQuickReplies])

  // Focus input
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => inputRef.current?.focus(), 150)
      return () => clearTimeout(timer)
    }
  }, [isOpen])

  const toggleSourceExpand = (msgId, srcIdx) => {
    const key = `${msgId}-${srcIdx}`
    setExpandedSources((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const toggleReasoningExpand = (msgId) => {
    setExpandedReasoning((prev) => ({ ...prev, [msgId]: !prev[msgId] }))
  }

  // Count known profile attributes
  const knownProfileCount = Object.values(farmerProfile).filter((v) => v !== null && v !== "").length

  // Start new conversation / reset session
  const handleStartNewChat = async () => {
    const newCid = `leafsense-${Math.random().toString(36).substring(2, 10)}`
    setConversationId(newCid)
    setConsultationStage("crop_context")
    setStageLabel("")
    setError(null)
    setLastFailedQuery("")
    const freshProfile = {
      crop: attachedPrediction ? getInitialCrop(attachedPrediction) : null,
      location: null,
      variety: null,
      plant_age: null,
      growth_stage: null,
      symptoms: null,
      first_noticed: null,
      is_spreading: null,
      affected_percentage: null,
      irrigation_type: null,
      recent_weather: null,
      treatments_applied: null,
    }
    setFarmerProfile(freshProfile)
    const freshWelcome = getInitialWelcomeMessage(attachedPrediction, chatLanguage)
    setMessages(freshWelcome)
    setActiveQuickReplies(freshWelcome[0].quickReplies || [])

    // Notify backend
    try {
      await fetch(`${API_BASE_URL}/api/agent/session/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId }),
      }).catch(() => {})
    } catch {
      // Ignored
    }
  }

  const handleSaveProfileEdit = () => {
    setCrop(editCrop.trim())
    setLocation(editLocation.trim())
    setFarmerProfile((prev) => ({
      ...prev,
      crop: editCrop.trim() || null,
      location: editLocation.trim() || null,
    }))
    setIsEditingProfile(false)
  }

  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || input || lastFailedQuery).trim()
    if (!query || loading) return

    setError(null)
    setInput("")

    const userMessage = {
      id: generateId("user"),
      sender: "farmer",
      text: query,
      time: "Just now",
    }

    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    // Build conversation history for multi-turn context
    const recentMessages = messages
      .slice(-6)
      .map((m) => ({
        role: m.sender === "farmer" ? "user" : "assistant",
        content: m.text,
      }))
    recentMessages.push({ role: "user", content: query })

    try {
      const payload = {
        question: query,
        crop: crop.trim() || farmerProfile.crop || undefined,
        location: location.trim() || farmerProfile.location || undefined,
        language: chatLanguage,
        top_k: 4,
        conversation_id: conversationId,
        messages: recentMessages,
        farmer_profile: farmerProfile,
      }

      if (attachedPrediction) {
        payload.detected_diseases = attachedPrediction.detectedDiseases
        payload.disease_predictions = {}
        attachedPrediction.allPredictions.forEach((p) => {
          payload.disease_predictions[p.label] = p.confidence
        })
      }

      const response = await fetch(`${API_BASE_URL}/api/agent/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        let errorMsg = `Server error ${response.status}`
        if (Array.isArray(errorData.detail)) {
          errorMsg = errorData.detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
        } else if (typeof errorData.detail === "string") {
          errorMsg = errorData.detail
        }
        throw new Error(errorMsg)
      }

      const data = await response.json()

      // Update multi-turn state
      if (data.conversation_id) setConversationId(data.conversation_id)
      if (data.stage) setConsultationStage(data.stage)
      if (data.stage_label) setStageLabel(data.stage_label)
      if (data.farmer_profile) {
        setFarmerProfile(data.farmer_profile)
        if (data.farmer_profile.crop && !crop) {
          setCrop(data.farmer_profile.crop)
          setEditCrop(data.farmer_profile.crop)
        }
        if (data.farmer_profile.location && !location) {
          setLocation(data.farmer_profile.location)
          setEditLocation(data.farmer_profile.location)
        }
      }

      const replies = data.quick_replies || data.follow_up_questions || []
      setActiveQuickReplies(replies)

      const agentMessage = {
        id: generateId("agent"),
        sender: "agent",
        text: data.answer || "No response generated.",
        grounded: data.grounded,
        sources: data.sources || [],
        confidenceAssessment: data.confidence_assessment,
        followUpQuestions: data.follow_up_questions || [],
        quickReplies: replies,
        disclaimer: data.disclaimer,
        modelUsed: data.model_used,
        stage: data.stage,
        stageLabel: data.stage_label,
        intents: data.intents || [],
        primaryIntent: data.primary_intent || "general_knowledge",
        differentialAnalysis: data.differential_analysis || null,
        evidenceType: data.evidence_type || "agricultural_reference",
        reasoningSummary: data.reasoning_summary || "",
        time: "Just now",
      }

      setMessages((prev) => [...prev, agentMessage])
      setLastFailedQuery("")
    } catch (err) {
      setLastFailedQuery(query)
      let displayError = err.message || "Failed to connect to the smart farming agent."
      if (displayError.includes("Failed to fetch") || displayError.includes("NetworkError")) {
        displayError = `Cannot connect to LeafSense backend at ${API_BASE_URL}. Ensure server is running.`
      }
      setError(displayError)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const isHi = chatLanguage === "hi"
  const stageInfo = STAGE_PROGRESS[consultationStage] || STAGE_PROGRESS.crop_context
  const currentStageText = stageLabel || (isHi ? stageInfo.label_hi : stageInfo.label_en)

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-end p-0 sm:p-6 bg-black/40 backdrop-blur-sm transition-opacity duration-200"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className={`w-full sm:w-[520px] h-[94vh] sm:h-[720px] max-h-[96vh] flex flex-col rounded-t-3xl sm:rounded-2xl shadow-2xl border transition-all duration-300 overflow-hidden ${
          darkMode
            ? "bg-gray-900 border-gray-700 text-gray-100"
            : "bg-white border-emerald-200 text-gray-900"
        }`}
      >
        {/* Header */}
        <div
          className={`px-4 py-3 border-b flex items-center justify-between flex-shrink-0 ${
            darkMode
              ? "bg-gray-800/90 border-gray-700"
              : "bg-gradient-to-r from-emerald-600 via-teal-700 to-emerald-800 text-white border-emerald-600"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-white/20 backdrop-blur-md flex items-center justify-center text-white shadow-inner relative">
              <Bot size={22} className="text-emerald-300" />
              <span className="w-2 h-2 rounded-full bg-emerald-400 absolute top-1 right-1 animate-ping"></span>
              <span className="w-2 h-2 rounded-full bg-emerald-400 absolute top-1 right-1"></span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-sm sm:text-base leading-tight">
                  {isHi ? "लीफसेंस एआई परामर्श" : "LeafSense Smart Assistant"}
                </h3>
                <span className="text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-emerald-500/30 text-emerald-100 border border-emerald-400/40">
                  Granite 4
                </span>
              </div>
              <p className="text-[11px] opacity-80 leading-tight">
                {isHi ? "इंटरएक्टिव किसान परामर्श एवं आरएजी ज्ञानकोश" : "Interactive Consultation & RAG Knowledge Base"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {/* New Consultation / Reset Session Button */}
            <button
              onClick={handleStartNewChat}
              className={`p-1.5 rounded-lg text-xs font-medium flex items-center gap-1 transition-all ${
                darkMode
                  ? "bg-gray-700 text-emerald-300 hover:bg-gray-600"
                  : "bg-white/20 text-white hover:bg-white/30"
              }`}
              title={isHi ? "नया परामर्श शुरू करें (चैट रीसेट)" : "Start New Consultation"}
            >
              <RotateCcw size={13} />
              <span className="hidden sm:inline text-[11px]">{isHi ? "नया चैट" : "New Chat"}</span>
            </button>

            {/* Language Toggle */}
            <button
              onClick={() => setChatLanguage(chatLanguage === "en" ? "hi" : "en")}
              className={`p-1.5 rounded-lg text-xs font-bold flex items-center gap-1 transition-all ${
                darkMode
                  ? "bg-gray-700 text-emerald-300 hover:bg-gray-600"
                  : "bg-white/20 text-white hover:bg-white/30"
              }`}
              title="Toggle Language (English / हिंदी)"
            >
              <Languages size={14} />
              <span>{chatLanguage === "en" ? "हिं" : "EN"}</span>
            </button>

            {/* Close Button */}
            <button
              onClick={onClose}
              className={`p-1.5 rounded-lg transition-all ${
                darkMode ? "hover:bg-gray-700 text-gray-300" : "hover:bg-white/20 text-white"
              }`}
              aria-label="Close assistant"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Consultation Progress Tracker Bar */}
        <div
          className={`px-3 py-1.5 border-b flex-shrink-0 flex items-center justify-between text-[11px] ${
            darkMode ? "bg-gray-800/60 border-gray-700 text-gray-300" : "bg-emerald-50/80 border-emerald-100 text-emerald-900"
          }`}
        >
          <div className="flex items-center gap-2 flex-1 mr-3">
            <span className="font-semibold whitespace-nowrap text-emerald-700 dark:text-emerald-300 flex items-center gap-1">
              <Sprout size={13} className="text-emerald-600" />
              {currentStageText}
            </span>
            <div className="flex-1 bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                style={{ width: `${stageInfo.percent}%` }}
              ></div>
            </div>
          </div>

          <button
            onClick={() => setShowProfileSummary(!showProfileSummary)}
            className="text-[10px] px-2 py-0.5 rounded-md font-medium border border-emerald-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 flex items-center gap-1"
          >
            <span>{isHi ? `विवरण (${knownProfileCount}/12)` : `Profile (${knownProfileCount}/12)`}</span>
            {showProfileSummary ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
        </div>

        {/* Collapsible Farmer Information Summary Panel */}
        {showProfileSummary && (
          <div
            className={`px-3 py-2.5 border-b text-xs flex-shrink-0 transition-all ${
              darkMode ? "bg-gray-800/80 border-gray-700 text-gray-200" : "bg-emerald-50/90 border-emerald-200 text-gray-800"
            }`}
          >
            <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-emerald-200 dark:border-gray-700">
              <span className="font-bold text-[11px] uppercase tracking-wider text-emerald-800 dark:text-emerald-300">
                {isHi ? "📋 एकत्रित किसान एवं खेत विवरण" : "📋 Collected Farmer & Field Details"}
              </span>
              <button
                onClick={() => setIsEditingProfile(!isEditingProfile)}
                className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 hover:underline flex items-center gap-1"
              >
                <Edit3 size={11} />
                <span>{isEditingProfile ? (isHi ? "रद्द करें" : "Cancel") : (isHi ? "संपादित करें" : "Edit")}</span>
              </button>
            </div>

            {isEditingProfile ? (
              <div className="grid grid-cols-2 gap-2 pt-1">
                <div>
                  <label className="text-[10px] font-bold text-gray-500 block">
                    {isHi ? "फसल (Crop)" : "Crop Name"}
                  </label>
                  <input
                    type="text"
                    value={editCrop}
                    onChange={(e) => setEditCrop(e.target.value)}
                    placeholder="e.g. Tomato, Wheat"
                    className={`w-full px-2 py-1 text-xs rounded border ${
                      darkMode ? "bg-gray-900 border-gray-700 text-white" : "bg-white border-emerald-300"
                    }`}
                  />
                </div>
                <div>
                  <label className="text-[10px] font-bold text-gray-500 block">
                    {isHi ? "स्थान (Location)" : "District / State"}
                  </label>
                  <input
                    type="text"
                    value={editLocation}
                    onChange={(e) => setEditLocation(e.target.value)}
                    placeholder="e.g. Nashik, Maharashtra"
                    className={`w-full px-2 py-1 text-xs rounded border ${
                      darkMode ? "bg-gray-900 border-gray-700 text-white" : "bg-white border-emerald-300"
                    }`}
                  />
                </div>
                <div className="col-span-2 flex justify-end gap-1.5 pt-1">
                  <button
                    onClick={handleSaveProfileEdit}
                    className="px-2.5 py-1 rounded bg-emerald-600 text-white text-[11px] font-bold hover:bg-emerald-700 flex items-center gap-1"
                  >
                    <Check size={12} />
                    <span>{isHi ? "सहेजें" : "Save Changes"}</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 text-[11px]">
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "फसल" : "Crop"}</span>
                  <span className="font-semibold text-emerald-700 dark:text-emerald-300 truncate block">
                    {farmerProfile.crop || crop || (isHi ? "अज्ञात" : "Not specified")}
                  </span>
                </div>
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "स्थान" : "Location"}</span>
                  <span className="font-semibold truncate block">
                    {farmerProfile.location || location || (isHi ? "अज्ञात" : "Not specified")}
                  </span>
                </div>
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "अवस्था" : "Stage"}</span>
                  <span className="font-semibold truncate block">
                    {farmerProfile.growth_stage || (isHi ? "प्रगति पर" : "In progress")}
                  </span>
                </div>
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "शुरुआत" : "Onset"}</span>
                  <span className="font-semibold truncate block">
                    {farmerProfile.first_noticed || (isHi ? "पूछना बाकी" : "Pending")}
                  </span>
                </div>
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "फैलाव" : "Spread"}</span>
                  <span className="font-semibold truncate block">
                    {farmerProfile.is_spreading || (isHi ? "पूछना बाकी" : "Pending")}
                  </span>
                </div>
                <div className="p-1 rounded bg-white/70 dark:bg-gray-900/60 border border-emerald-100 dark:border-gray-700">
                  <span className="text-gray-400 block text-[9px] uppercase font-bold">{isHi ? "सिंचाई / मौसम" : "Irrigation"}</span>
                  <span className="font-semibold truncate block">
                    {farmerProfile.irrigation_type || farmerProfile.recent_weather || (isHi ? "पूछना बाकी" : "Pending")}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Attached Scan Context Header Bar */}
        {attachedPrediction && (
          <div
            className={`px-3 py-1.5 border-b text-xs flex items-center justify-between flex-shrink-0 ${
              darkMode ? "bg-emerald-950/40 border-gray-800" : "bg-emerald-50 border-emerald-100"
            }`}
          >
            <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300 font-semibold truncate text-[11px]">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="truncate">
                📸 {attachedPrediction.formatted} ({attachedPrediction.percentage})
              </span>
              <span className="text-[10px] text-gray-400 italic hidden sm:inline">
                ({isHi ? "प्रारंभिक स्कैन" : "Preliminary Scan"})
              </span>
            </div>
            <button
              onClick={() => setAttachedPrediction(null)}
              className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-red-100 hover:bg-red-200 dark:bg-gray-800 text-red-600 dark:text-red-400 font-medium flex items-center gap-0.5"
              title={isHi ? "स्कैन संदर्भ हटाएं" : "Detach scan context"}
            >
              <X size={10} />
              <span>{isHi ? "हटाएं" : "Detach"}</span>
            </button>
          </div>
        )}

        {/* Messages Stream */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, index) => {
            const isLastAgentMessage = msg.sender === "agent" && index === messages.length - 1
            return (
              <div
                key={msg.id}
                className={`flex flex-col ${msg.sender === "farmer" ? "items-end" : "items-start"}`}
              >
                {/* Message Bubble */}
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed max-w-[92%] shadow-sm ${
                    msg.sender === "farmer"
                      ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-tr-none"
                      : darkMode
                        ? "bg-gray-800 text-gray-100 rounded-tl-none border border-gray-700"
                        : "bg-white text-gray-900 rounded-tl-none border border-gray-200 shadow-sm"
                  }`}
                >
                  {/* Agent Header / Model Tag */}
                  {msg.sender === "agent" && (
                    <div className="flex flex-col gap-1 mb-2 pb-1.5 border-b border-gray-100 dark:border-gray-700/60">
                      <div className="flex items-center gap-1.5 text-[11px] font-bold text-emerald-600 dark:text-emerald-400">
                        <Sparkles size={12} />
                        <span>IBM Granite 4 Farm Advisor</span>
                        <span className="text-gray-400 ml-auto font-normal text-[10px]">
                          {msg.time}
                        </span>
                      </div>

                      {/* Evidence Type & Intent Badges */}
                      {(msg.evidenceType || (msg.intents && msg.intents.length > 0)) && (
                        <div className="flex flex-wrap items-center gap-1 pt-0.5">
                          {msg.evidenceType === "official_guidance" && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 flex items-center gap-0.5">
                              <Award size={9} />
                              <span>{isHi ? "सरकारी / ICAR / KVK संस्तुति" : "ICAR / KVK Official Guidance"}</span>
                            </span>
                          )}
                          {msg.evidenceType === "verified_research" && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 flex items-center gap-0.5">
                              <CheckCircle2 size={9} />
                              <span>{isHi ? "सत्यापित कृषि शोध" : "Verified Agronomy Research"}</span>
                            </span>
                          )}
                          {msg.evidenceType === "agricultural_reference" && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 flex items-center gap-0.5">
                              <FileText size={9} />
                              <span>{isHi ? "कृषि संदर्भ ज्ञानकोश" : "Agricultural Knowledge Base"}</span>
                            </span>
                          )}
                          {msg.evidenceType === "general_agronomy" && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300 flex items-center gap-0.5">
                              <Sprout size={9} />
                              <span>{isHi ? "सामान्य कृषि मार्गदर्शन" : "General Agronomy Guidance"}</span>
                            </span>
                          )}
                          {msg.intents && msg.intents.length > 1 && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300">
                              {isHi ? `बहु-विषयक (${msg.intents.length})` : `Multi-topic (${msg.intents.length})`}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Message Body */}
                  <div className="whitespace-pre-line space-y-2 leading-relaxed">
                    {msg.text}
                  </div>

                  {/* Differential Disease Comparison Card */}
                  {msg.differentialAnalysis && (
                    <div
                      className={`mt-3 p-3 rounded-xl border text-xs ${
                        darkMode
                          ? "bg-emerald-950/20 border-emerald-800/80 text-gray-200"
                          : "bg-emerald-50/70 border-emerald-200 text-gray-800"
                      }`}
                    >
                      <div className="flex items-center gap-1.5 font-bold text-emerald-800 dark:text-emerald-300 mb-2 pb-1 border-b border-emerald-200 dark:border-emerald-800/60">
                        <GitCompare size={14} className="text-emerald-600 dark:text-emerald-400" />
                        <span>{isHi ? "तुलनात्मक रोग विश्लेषण" : "Differential Disease Analysis"}</span>
                        <span className="text-[10px] ml-auto font-medium opacity-80 truncate max-w-[170px]">
                          {msg.differentialAnalysis.disease_a} vs {msg.differentialAnalysis.disease_b}
                        </span>
                      </div>

                      {msg.differentialAnalysis.shared_symptoms && (
                        <div className="mb-2 text-[11px] bg-white/80 dark:bg-gray-900/60 p-2 rounded-lg border border-emerald-100 dark:border-gray-700">
                          <span className="font-semibold text-amber-700 dark:text-amber-400">
                            {isHi ? "सांझे लक्षण: " : "Shared Symptoms: "}
                          </span>
                          <span>{msg.differentialAnalysis.shared_symptoms}</span>
                        </div>
                      )}

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
                        <div className="p-2 rounded-lg bg-white/90 dark:bg-gray-900/70 border border-emerald-200 dark:border-gray-700">
                          <span className="font-bold text-emerald-700 dark:text-emerald-300 block text-[11px] mb-1">
                            🌿 {msg.differentialAnalysis.disease_a}
                          </span>
                          <p className="text-[10px] text-gray-600 dark:text-gray-300 leading-snug">
                            {msg.differentialAnalysis.disease_a_markers}
                          </p>
                        </div>

                        <div className="p-2 rounded-lg bg-white/90 dark:bg-gray-900/70 border border-emerald-200 dark:border-gray-700">
                          <span className="font-bold text-teal-700 dark:text-teal-300 block text-[11px] mb-1">
                            🌿 {msg.differentialAnalysis.disease_b}
                          </span>
                          <p className="text-[10px] text-gray-600 dark:text-gray-300 leading-snug">
                            {msg.differentialAnalysis.disease_b_markers}
                          </p>
                        </div>
                      </div>

                      {msg.differentialAnalysis.target_probe && (
                        <div className="text-[11px] bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/70 p-2 rounded-lg text-amber-900 dark:text-amber-200">
                          <span className="font-bold block text-[10px] uppercase tracking-wider text-amber-800 dark:text-amber-400">
                            🔎 {isHi ? "खेत में अंतर पहचानने की जांच:" : "Field Verification Probe:"}
                          </span>
                          <p className="mt-0.5 leading-snug">{msg.differentialAnalysis.target_probe}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Confidence Assessment Pill */}
                  {msg.confidenceAssessment && (
                    <div
                      className={`mt-3 p-2 rounded-lg text-xs border flex items-start gap-1.5 ${
                        darkMode
                          ? "bg-gray-900/60 border-emerald-900/60 text-emerald-300"
                          : "bg-emerald-50 border-emerald-200 text-emerald-800"
                      }`}
                    >
                      <ShieldCheck size={14} className="flex-shrink-0 mt-0.5 text-emerald-600" />
                      <div>
                        <span className="font-bold">
                          {isHi ? "एआई निश्चितता मूल्यांकन: " : "Certainty Assessment: "}
                        </span>
                        <span>{msg.confidenceAssessment}</span>
                      </div>
                    </div>
                  )}

                  {/* Reasoning & Evidence Rationale Accordion */}
                  {msg.reasoningSummary && (
                    <div className="mt-2.5 rounded-lg border border-purple-200 dark:border-purple-900/60 bg-purple-50/50 dark:bg-purple-950/20 text-xs overflow-hidden">
                      <button
                        onClick={() => toggleReasoningExpand(msg.id)}
                        className="w-full px-2.5 py-1.5 flex items-center justify-between text-purple-800 dark:text-purple-300 font-semibold hover:bg-purple-100/40 dark:hover:bg-purple-900/40 text-[11px] transition-colors"
                      >
                        <div className="flex items-center gap-1.5">
                          <Brain size={13} className="text-purple-600 dark:text-purple-400" />
                          <span>{isHi ? "तार्किक विश्लेषण एवं साक्ष्य विवरण" : "Reasoning & Evidence Rationale"}</span>
                        </div>
                        <div className="flex items-center gap-1 text-[10px]">
                          <span className="opacity-75">
                            {expandedReasoning[msg.id] ? (isHi ? "छिपाएं" : "Hide") : (isHi ? "देखें" : "View")}
                          </span>
                          {expandedReasoning[msg.id] ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                        </div>
                      </button>
                      {expandedReasoning[msg.id] && (
                        <div className="px-2.5 py-2 border-t border-purple-200 dark:border-purple-900/60 bg-white/70 dark:bg-gray-900/70 text-[11px] text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-line">
                          {msg.reasoningSummary}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Grounding & Citations Section */}
                  {msg.sender === "agent" && msg.grounded !== undefined && (
                    <div className="mt-3 pt-2.5 border-t border-gray-100 dark:border-gray-700/70">
                      {msg.grounded ? (
                        <div className="space-y-2">
                          <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-300">
                            <FileText size={13} />
                            <span>
                              {isHi
                                ? `सत्यापित कृषि स्रोत (${msg.sources.length})`
                                : `Verified Reference Citations (${msg.sources.length})`}
                            </span>
                          </div>

                          {/* Sources Cards */}
                          <div className="space-y-1.5">
                            {msg.sources.map((src, idx) => {
                              const isExpanded = expandedSources[`${msg.id}-${idx}`]
                              return (
                                <div
                                  key={idx}
                                  className={`rounded-lg p-2 text-xs border transition-all ${
                                    darkMode
                                      ? "bg-gray-900/50 border-gray-700/80 hover:border-emerald-600"
                                      : "bg-gray-50 border-gray-200 hover:border-emerald-400"
                                  }`}
                                >
                                  <div
                                    className="flex items-center justify-between cursor-pointer"
                                    onClick={() => toggleSourceExpand(msg.id, idx)}
                                  >
                                    <span className="font-semibold text-gray-800 dark:text-gray-200 truncate pr-2">
                                      📄 {src.document} (P. {src.page})
                                    </span>
                                    <div className="flex items-center gap-1 flex-shrink-0">
                                      {src.authority_type === "official_guidance" && (
                                        <span className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 flex items-center gap-0.5 border border-emerald-300 dark:border-emerald-700">
                                          <Award size={9} />
                                          <span>{isHi ? "सरकारी/KVK" : "Official/KVK"}</span>
                                        </span>
                                      )}
                                      {src.authority_type === "verified_research" && (
                                        <span className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 flex items-center gap-0.5 border border-blue-300 dark:border-blue-700">
                                          <CheckCircle2 size={9} />
                                          <span>{isHi ? "शोध" : "Research"}</span>
                                        </span>
                                      )}
                                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 font-bold">
                                        {Math.round(src.relevance_score * 100)}% match
                                      </span>
                                      {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                    </div>
                                  </div>

                                  {isExpanded && (
                                    <p className="mt-1.5 text-[11px] text-gray-600 dark:text-gray-300 pl-2 border-l-2 border-emerald-500 italic bg-white/40 dark:bg-gray-800/40 p-1 rounded">
                                      "{src.snippet}"
                                    </p>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      ) : (
                        <div
                          className={`p-2 rounded-lg text-xs flex items-center gap-1.5 border ${
                            darkMode
                              ? "bg-amber-950/30 border-amber-900/50 text-amber-300"
                              : "bg-amber-50 border-amber-200 text-amber-800"
                          }`}
                        >
                          <AlertTriangle size={14} className="flex-shrink-0 text-amber-500" />
                          <span>
                            {isHi
                              ? "ज्ञानकोश में कोई सत्यापित संदर्भ दस्तावेज नहीं मिला। सलाह सामान्य कृषि सिद्धांतों पर आधारित है।"
                              : "No matching reference document found in knowledge base. Response is general educational guidance."}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Inline Dynamic Quick-Reply Chips (rendered under the message) */}
                  {isLastAgentMessage && activeQuickReplies && activeQuickReplies.length > 0 && !loading && (
                    <div className="mt-3 pt-2.5 border-t border-gray-100 dark:border-gray-700/60">
                      <p className="text-[11px] font-bold text-gray-500 dark:text-gray-400 mb-1.5 flex items-center gap-1">
                        <span>👉</span>
                        <span>{isHi ? "त्वरित उत्तर / आगे की बात:" : "Quick Replies / Continue:"}</span>
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {activeQuickReplies.map((replyText, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSendMessage(replyText)}
                            className={`text-[11px] text-left px-3 py-1.5 rounded-full border transition-all font-medium active:scale-95 ${
                              replyText.includes("Skip") || replyText.includes("छोड़ें")
                                ? "bg-amber-50 hover:bg-amber-100 dark:bg-amber-950/40 border-amber-300 text-amber-800 dark:text-amber-300 font-bold"
                                : darkMode
                                  ? "bg-gray-700/80 hover:bg-emerald-950 border-gray-600 text-emerald-300 hover:border-emerald-500"
                                  : "bg-emerald-50 hover:bg-emerald-100 border-emerald-300 text-emerald-900 shadow-xs"
                            }`}
                          >
                            {replyText}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Disclaimer (shown subtly) */}
                  {msg.disclaimer && (
                    <p className="mt-2 text-[10px] text-gray-400 dark:text-gray-500 italic leading-snug">
                      {msg.disclaimer}
                    </p>
                  )}
                </div>

                {msg.sender === "farmer" && (
                  <span className="text-[10px] text-gray-400 mt-1 mr-1">
                    {msg.time}
                  </span>
                )}
              </div>
            )
          })}

          {/* Loading Indicator */}
          {loading && (
            <div className="flex items-start gap-2">
              <div
                className={`rounded-2xl rounded-tl-none px-4 py-3 text-xs border shadow-sm flex items-center gap-2 ${
                  darkMode
                    ? "bg-gray-800 border-gray-700 text-gray-200"
                    : "bg-white border-emerald-200 text-gray-800"
                }`}
              >
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-emerald-500 border-t-transparent"></div>
                <span>
                  {isHi
                    ? "आईबीएम ग्रेनाइट 4 और कृषि ज्ञानकोश से परामर्श लिया जा रहा है..."
                    : "Consulting IBM Granite 4 & Agriculture Knowledge Base..."}
                </span>
              </div>
            </div>
          )}

          {/* Error Alert with Working Retry */}
          {error && (
            <div className="p-3 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs flex items-center justify-between">
              <div className="flex items-center gap-1.5 pr-2">
                <AlertTriangle size={14} className="flex-shrink-0" />
                <span>{error}</span>
              </div>
              <button
                onClick={() => handleSendMessage(lastFailedQuery)}
                className="px-2.5 py-1 rounded bg-red-600 text-white font-bold text-[10px] hover:bg-red-700 flex items-center gap-1 flex-shrink-0 active:scale-95"
              >
                <RefreshCw size={10} />
                <span>{isHi ? "पुनः प्रयास करें" : "Retry"}</span>
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div
          className={`p-3 border-t flex-shrink-0 ${
            darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              maxLength={1000}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              placeholder={
                isHi
                  ? "फसल रोग, लक्षण, फैलाव या उपचार के बारे में पूछें..."
                  : "Type your observation, question, or response..."
              }
              className={`flex-1 px-3.5 py-2.5 text-xs sm:text-sm rounded-xl border focus:outline-none transition-all ${
                darkMode
                  ? "bg-gray-900 border-gray-700 text-gray-100 placeholder-gray-500 focus:border-emerald-500"
                  : "bg-gray-50 border-emerald-200 text-gray-900 placeholder-gray-400 focus:bg-white focus:border-emerald-500"
              }`}
            />

            <button
              onClick={() => handleSendMessage()}
              disabled={loading || !input.trim()}
              className={`p-2.5 rounded-xl transition-all flex items-center justify-center ${
                loading || !input.trim()
                  ? "bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed"
                  : "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-md active:scale-95"
              }`}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
