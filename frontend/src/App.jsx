import { useState, useEffect, useRef } from "react"
import { translations } from "./translations"
import html2canvas from "html2canvas"
import CameraCapture from "./CameraCapture"
import ChatAssistant from "./ChatAssistant"
import {
  Sun,
  Moon,
  Sprout,
  Zap,
  Bot,
  Globe,
  Camera,
  Image as ImageIcon,
  Trophy,
  Medal,
  Lightbulb,
  Search,
  Download,
  Trash2,
  AlertCircle,
  Layers,
  LayoutGrid,
} from "lucide-react"

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

export default function App() {
  const [uiLanguage, setUiLanguage] = useState("en")
  const [darkMode, setDarkMode] = useState(false)
  const [predictions, setPredictions] = useState([])
  const [detectedDiseases, setDetectedDiseases] = useState([])
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [chatContext, setChatContext] = useState(null)
  const t = translations[uiLanguage] || translations.en

  const handleOpenChat = (customPredictions = null, customDetected = null) => {
    const activePreds = customPredictions || predictions
    const activeDetected =
      customDetected ||
      (detectedDiseases && detectedDiseases.length > 0
        ? detectedDiseases
        : activePreds ? activePreds.map((p) => p.label) : [])

    if (activePreds && activePreds.length > 0) {
      setChatContext({
        predictions: activePreds,
        detectedDiseases: activeDetected,
      })
    } else {
      setChatContext(null)
    }
    setIsChatOpen(true)
  }

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [darkMode])

  const languages = [
    { code: "en", name: "English", flag: "🇬🇧" },
    { code: "hi", name: "हिंदी", flag: "🇮🇳" },
    { code: "bn", name: "বাংলা", flag: "🇧🇩" },
    { code: "ta", name: "தமிழ்", flag: "🇮🇳" },
    { code: "te", name: "తెలుగు", flag: "🇮🇳" },
    { code: "mr", name: "मराठी", flag: "🇮🇳" },
    { code: "gu", name: "ગુજરાતી", flag: "🇮🇳" },
    { code: "pa", name: "ਪੰਜਾਬੀ", flag: "🇮🇳" },
    { code: "ml", name: "മലയാളം", flag: "🇮🇳" },
    { code: "kn", name: "ಕನ್ನಡ", flag: "🇮🇳" },
    { code: "or", name: "ଓଡ଼ିଆ", flag: "🇮🇳" },
  ]

  return (
    <div
      className={`min-h-screen ${
        darkMode
          ? "bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 text-gray-100"
          : "bg-gradient-to-br from-green-50 via-emerald-50 to-teal-100 text-gray-900"
      } flex flex-col items-center p-2 sm:p-4 md:p-6 transition-colors duration-300`}
    >
      <header className="w-full max-w-7xl mb-4 sm:mb-6">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-0">
          <div className="text-center sm:text-left flex-1 w-full sm:w-auto">
            <h1
              className={`text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r ${
                darkMode ? "from-green-400 to-emerald-300" : "from-green-700 to-emerald-600"
              }`}
            >
              🌿 {t.appTitle}
            </h1>
            <p
              className={`mt-1 sm:mt-2 ${
                darkMode ? "text-gray-300" : "text-gray-600"
              } text-xs sm:text-sm md:text-base font-medium`}
            >
              {t.appSubtitle}
            </p>
          </div>

          {/* Controls: AI Assistant, Dark Mode & Language */}
          <div className="flex gap-2 items-center w-full sm:w-auto justify-center sm:justify-end">
            <button
              onClick={() => handleOpenChat()}
              className={`px-3 py-2 sm:px-3.5 sm:py-2 rounded-xl transition-all duration-300 font-semibold text-xs sm:text-sm flex items-center gap-1.5 shadow-md active:scale-95 ${
                darkMode
                  ? "bg-gradient-to-r from-emerald-700 to-teal-800 text-emerald-100 hover:from-emerald-600 hover:to-teal-700 border border-emerald-600"
                  : "bg-gradient-to-r from-emerald-600 to-teal-600 text-white hover:from-emerald-700 hover:to-teal-700 shadow-emerald-200"
              }`}
              title={t.chatButton || "Ask AI Assistant"}
            >
              <Bot size={18} />
              <span className="hidden sm:inline">{t.chatButton || "AI Assistant"}</span>
            </button>

            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 sm:p-2.5 rounded-xl transition-all duration-300 ${
                darkMode
                  ? "bg-gray-800 text-yellow-400 hover:bg-gray-700 border border-gray-700"
                  : "bg-white text-gray-700 hover:bg-gray-100 border border-green-200"
              } shadow-md`}
              title={darkMode ? t.lightMode : t.darkMode}
              aria-label={darkMode ? t.lightMode : t.darkMode}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>

            <select
              value={uiLanguage}
              onChange={(e) => setUiLanguage(e.target.value)}
              className={`p-2 sm:p-2.5 border-2 rounded-xl focus:outline-none transition-all text-xs sm:text-sm font-semibold flex-1 sm:flex-initial shadow-md ${
                darkMode
                  ? "bg-gray-800 border-gray-700 text-gray-100 focus:border-green-400"
                  : "bg-white border-green-300 text-gray-900 focus:border-green-600"
              }`}
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.flag} {lang.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      <main
        className={`w-full max-w-7xl rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden transition-all duration-300 border ${
          darkMode ? "bg-gray-900 border-gray-800" : "bg-white border-green-100"
        }`}
      >
        <div className="grid lg:grid-cols-5">
          {/* Left Hero Sidebar */}
          <div
            className={`lg:col-span-2 bg-gradient-to-br ${
              darkMode ? "from-emerald-900 to-teal-950" : "from-emerald-700 to-teal-800"
            } text-white p-6 sm:p-8 md:p-10 flex flex-col justify-between ${
              predictions.length > 0 ? "hidden lg:flex" : ""
            }`}
          >
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md text-xs font-semibold mb-4 text-emerald-200 border border-white/10">
                <span>AI-Powered Pathology</span>
              </div>
              <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold leading-tight">
                {t.mainHeading}
              </h2>
              <p className="mt-3 text-sm sm:text-base text-emerald-100/90 leading-relaxed">
                {t.mainDescription}
              </p>

              <div className="mt-8 space-y-3">
                <div className="bg-white/10 p-3.5 rounded-xl backdrop-blur-sm flex items-start gap-3 border border-white/10">
                  <Sprout size={24} className="flex-shrink-0 text-emerald-300" />
                  <div>
                    <h3 className="font-semibold text-sm sm:text-base">{t.feature1Title}</h3>
                    <p className="text-xs text-emerald-100/80">{t.feature1Desc}</p>
                  </div>
                </div>

                <div className="bg-white/10 p-3.5 rounded-xl backdrop-blur-sm flex items-start gap-3 border border-white/10">
                  <Zap size={24} className="flex-shrink-0 text-amber-300" />
                  <div>
                    <h3 className="font-semibold text-sm sm:text-base">{t.feature2Title}</h3>
                    <p className="text-xs text-emerald-100/80">{t.feature2Desc}</p>
                  </div>
                </div>

                <div className="bg-white/10 p-3.5 rounded-xl backdrop-blur-sm flex items-start gap-3 border border-white/10">
                  <Bot size={24} className="flex-shrink-0 text-cyan-300" />
                  <div>
                    <h3 className="font-semibold text-sm sm:text-base">{t.feature3Title}</h3>
                    <p className="text-xs text-emerald-100/80">{t.feature3Desc}</p>
                  </div>
                </div>

                <div className="bg-white/10 p-3.5 rounded-xl backdrop-blur-sm flex items-start gap-3 border border-white/10">
                  <Globe size={24} className="flex-shrink-0 text-teal-200" />
                  <div>
                    <h3 className="font-semibold text-sm sm:text-base">{t.feature4Title}</h3>
                    <p className="text-xs text-emerald-100/80">{t.feature4Desc}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 pt-4 border-t border-white/10 text-xs text-emerald-200/70">
              Covers Apple, Tomato, Grape, Corn & Potato crops.
            </div>
          </div>

          {/* Right Upload & Diagnosis Pane */}
          <div className="lg:col-span-3">
            <UploadSection
              uiLanguage={uiLanguage}
              darkMode={darkMode}
              predictions={predictions}
              setPredictions={setPredictions}
              detectedDiseases={detectedDiseases}
              setDetectedDiseases={setDetectedDiseases}
              onAskAi={handleOpenChat}
            />
          </div>
        </div>
      </main>

      <footer
        className={`mt-6 mb-4 text-center text-xs px-4 max-w-4xl space-y-1.5 ${
          darkMode ? "text-gray-400" : "text-gray-600"
        }`}
      >
        <p className="font-medium">{t.footer}</p>
        <p className="text-[11px] opacity-80">
          Assistive AI tool for preliminary diagnostics. Always confirm with approved chemical labels and local agricultural extension (KVK) specialists before pesticide application.
        </p>
        <p className="text-[11px] pt-1">
          <a
            href="https://github.com/RudraMandloi834/Leafsense"
            target="_blank"
            rel="noopener noreferrer"
            className="text-emerald-600 dark:text-emerald-400 hover:underline font-semibold"
          >
            LeafSense Open Source Repository on GitHub
          </a>
        </p>
      </footer>

      {/* Floating Assistant Trigger Button */}
      {!isChatOpen && (
        <button
          onClick={() => handleOpenChat()}
          className="fixed bottom-5 right-5 sm:bottom-6 sm:right-6 z-40 p-3.5 sm:px-4 sm:py-3 rounded-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-2xl flex items-center gap-2.5 transition-all duration-300 hover:scale-105 active:scale-95 border border-white/20"
          aria-label="Open AI Assistant"
        >
          <div className="relative">
            <Bot size={22} />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-emerald-300 rounded-full animate-ping"></span>
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-emerald-400 rounded-full"></span>
          </div>
          <span className="hidden sm:inline font-bold text-sm tracking-wide">
            {t.chatButton || "Ask AI Assistant"}
          </span>
        </button>
      )}

      {/* Reusable Chat Assistant Modal / Drawer */}
      <ChatAssistant
        key={chatContext?.predictions?.[0]?.label || (isChatOpen ? "open" : "closed")}
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        initialContext={chatContext}
        uiLanguage={uiLanguage}
        darkMode={darkMode}
      />
    </div>
  )
}

function UploadSection({
  uiLanguage,
  darkMode,
  predictions,
  setPredictions,
  detectedDiseases: externalDetected,
  setDetectedDiseases: externalSetDetected,
  onAskAi,
}) {
  const t = translations[uiLanguage] || translations.en
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [advice, setAdvice] = useState("")
  const [customLanguage, setCustomLanguage] = useState(null)
  const language = customLanguage || uiLanguage
  const [adviceLanguageOverride, setAdviceLanguageOverride] = useState(null)
  const adviceLanguage = adviceLanguageOverride || language
  const [localDetectedDiseases, setLocalDetectedDiseases] = useState([])
  const detectedDiseases = externalDetected || localDetectedDiseases
  const setDetectedDiseases = (val) => {
    setLocalDetectedDiseases(val)
    if (externalSetDetected) externalSetDetected(val)
  }
  const [error, setError] = useState("")
  const [dragActive, setDragActive] = useState(false)
  const [imageZoomed, setImageZoomed] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [qualityCheck, setQualityCheck] = useState(null)
  const [checkingQuality, setCheckingQuality] = useState(false)
  const [visualizations, setVisualizations] = useState([])
  const [combinedVisualization, setCombinedVisualization] = useState(null)
  const [showCombined, setShowCombined] = useState(true)

  const resultsRef = useRef(null)
  const fileInputRef = useRef(null)

  const languages = [
    { code: "en", name: "English", flag: "🇬🇧" },
    { code: "hi", name: "हिंदी", flag: "🇮🇳" },
    { code: "bn", name: "বাংলা", flag: "🇧🇩" },
    { code: "ta", name: "தமிழ்", flag: "🇮🇳" },
    { code: "te", name: "తెలుగు", flag: "🇮🇳" },
    { code: "mr", name: "मराठी", flag: "🇮🇳" },
    { code: "gu", name: "ગુજરાતી", flag: "🇮🇳" },
    { code: "pa", name: "ਪੰਜਾਬੀ", flag: "🇮🇳" },
    { code: "ml", name: "മലയാളം", flag: "🇮🇳" },
    { code: "kn", name: "ಕನ್ನಡ", flag: "🇮🇳" },
    { code: "or", name: "ଓଡ଼ିଆ", flag: "🇮🇳" },
  ]

  const handleFile = (selectedFile, previewUrl = null) => {
    if (!selectedFile) return

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file (JPG, PNG, WebP)")
      return
    }

    setFile(selectedFile)
    setPreview(previewUrl || URL.createObjectURL(selectedFile))
    setPredictions([])
    setAdvice("")
    setDetectedDiseases([])
    setVisualizations([])
    setCombinedVisualization(null)
    setError("")
    setQualityCheck(null)

    // Pre-flight image quality check
    checkImageQuality(selectedFile)
  }

  const checkImageQuality = async (imageFile) => {
    setCheckingQuality(true)
    setQualityCheck(null)

    try {
      const formData = new FormData()
      formData.append("file", imageFile)

      const response = await fetch(`${API_BASE_URL}/check-quality`, {
        method: "POST",
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setQualityCheck(data.quality)
        }
      }
    } catch {
      // Quality check failure is non-fatal; user can still proceed with prediction
    } finally {
      setCheckingQuality(false)
    }
  }

  const handleCameraCapture = (capturedFile, previewUrl) => {
    handleFile(capturedFile, previewUrl)
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const resetUpload = () => {
    setFile(null)
    setPreview(null)
    setPredictions([])
    setAdvice("")
    setDetectedDiseases([])
    setVisualizations([])
    setCombinedVisualization(null)
    setError("")
    setQualityCheck(null)
    setImageZoomed(false)
    setCustomLanguage(null)
    setAdviceLanguageOverride(null)
  }

  // Export diagnostic report
  const downloadReport = async () => {
    if (!resultsRef.current) return
    try {
      const canvas = await html2canvas(resultsRef.current, {
        backgroundColor: darkMode ? "#111827" : "#ffffff",
        scale: 2,
      })
      const link = document.createElement("a")
      link.download = `leafsense-diagnostic-report-${Date.now()}.png`
      link.href = canvas.toDataURL()
      link.click()
    } catch (err) {
      console.error("Export report error:", err)
    }
  }

  const getSeverity = (confidence) => {
    if (confidence >= 0.7) return { level: t.high, colorClass: "text-red-500", stroke: "#ef4444" }
    if (confidence >= 0.45) return { level: t.medium, colorClass: "text-amber-500", stroke: "#f59e0b" }
    return { level: t.low, colorClass: "text-emerald-500", stroke: "#10b981" }
  }

  const CircularProgress = ({ percentage, strokeColor = "#10b981", size = 96, strokeWidth = 8 }) => {
    const radius = (size - strokeWidth) / 2
    const circumference = radius * 2 * Math.PI
    const offset = circumference - (percentage / 100) * circumference

    return (
      <div className="relative inline-flex items-center justify-center">
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={darkMode ? "#374151" : "#e5e7eb"}
            strokeWidth={strokeWidth}
            fill="none"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-xl font-extrabold ${darkMode ? "text-gray-100" : "text-gray-900"}`}>
            {percentage.toFixed(1)}%
          </span>
        </div>
      </div>
    )
  }

  const translateAdvice = async (newLanguage) => {
    if (!detectedDiseases.length || newLanguage === adviceLanguage) return

    setTranslating(true)
    setError("")

    try {
      const response = await fetch(`${API_BASE_URL}/translate-advice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diseases: detectedDiseases,
          target_language: newLanguage,
        }),
      })

      if (!response.ok) {
        throw new Error(`Translation status: ${response.status}`)
      }

      const data = await response.json()
      if (data.success) {
        setAdvice(data.advice)
        setAdviceLanguageOverride(newLanguage)
      } else {
        setError(`Translation error: ${data.error}`)
      }
    } catch (err) {
      setError(`Translation failed: ${err.message}`)
    } finally {
      setTranslating(false)
    }
  }

  const predictDisease = async () => {
    if (!file) return

    setLoading(true)
    setError("")
    setPredictions([])
    setAdvice("")
    setVisualizations([])
    setCombinedVisualization(null)
    setShowCombined(true)

    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("language", language)

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errDetail = await response.text()
        throw new Error(`Server returned error ${response.status}: ${errDetail}`)
      }

      const data = await response.json()

      if (!data.predictions) {
        setError("Invalid response format received from inference server")
        return
      }

      const preds = Object.entries(data.predictions).map(([label, confidence]) => ({
        label,
        confidence,
      }))

      setPredictions(preds)
      setAdvice(data.advice || "")
      setAdviceLanguageOverride(language)
      setDetectedDiseases(data.detected_diseases || [])

      if (data.visualizations && data.visualizations.length > 0) {
        setVisualizations(data.visualizations)
      }
      if (data.combined_visualization) {
        setCombinedVisualization(data.combined_visualization)
      }
    } catch (err) {
      setError(`Connection failure: ${err.message}. Ensure backend is active at ${API_BASE_URL}.`)
    } finally {
      setLoading(false)
    }
  }

  // Safe and clean advisory markdown parser
  const renderAdvice = (text) => {
    if (!text) return null

    const lines = text.split("\n")
    const elements = []
    let key = 0

    const headerEmojis = ["🌾", "⚠️", "🔍", "🎯", "🍎", "🌿"]
    const subheaderEmojis = ["🦠", "🍃", "💊", "🛡️", "💧", "👨‍🌾", "🗑️"]

    lines.forEach((line) => {
      const trimmed = line.trim()
      if (!trimmed) return

      const isHeaderEmoji = headerEmojis.some((e) => trimmed.startsWith(e))
      const isSubheaderEmoji = subheaderEmojis.some((e) => trimmed.startsWith(e))

      if (trimmed.startsWith("## ") || isHeaderEmoji) {
        elements.push(
          <h3
            key={key++}
            className={`text-lg sm:text-xl font-bold mt-6 mb-3 pb-2 border-b-2 flex items-center gap-2 ${
              darkMode ? "text-emerald-300 border-emerald-800" : "text-emerald-800 border-emerald-200"
            }`}
          >
            {trimmed.replace(/^##\s*/, "")}
          </h3>
        )
      } else if (trimmed.startsWith("### ") || isSubheaderEmoji) {
        elements.push(
          <h4
            key={key++}
            className={`text-sm sm:text-base font-semibold mt-4 mb-2 p-2 rounded-lg ${
              darkMode ? "text-gray-100 bg-gray-800/80 border border-gray-700" : "text-gray-900 bg-emerald-50 border border-emerald-100"
            }`}
          >
            {trimmed.replace(/^###\s*/, "")}
          </h4>
        )
      } else if (trimmed === "---") {
        elements.push(
          <hr
            key={key++}
            className={`my-6 border-t ${darkMode ? "border-gray-700" : "border-emerald-200"}`}
          />
        )
      } else if (trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
        elements.push(
          <li
            key={key++}
            className={`leading-relaxed mb-1.5 ml-5 text-sm list-disc ${
              darkMode ? "text-gray-300" : "text-gray-700"
            }`}
          >
            {trimmed.replace(/^[-•]\s*/, "")}
          </li>
        )
      } else {
        elements.push(
          <p
            key={key++}
            className={`leading-relaxed mb-2 text-sm ${darkMode ? "text-gray-300" : "text-gray-700"}`}
          >
            {trimmed}
          </p>
        )
      }
    })

    return <div className="space-y-1">{elements}</div>
  }

  return (
    <div className={`p-4 sm:p-6 md:p-8 flex flex-col ${darkMode ? "text-gray-100" : "text-gray-900"}`}>
      {/* Upload Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 sm:mb-6 gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight">
            {t.uploadTitle}
          </h2>
          <p className={`text-xs sm:text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            Upload or capture high-resolution foliage for diagnostic analysis
          </p>
        </div>

        {preview && (
          <div className="flex gap-2 w-full sm:w-auto">
            {predictions.length > 0 && (
              <button
                onClick={downloadReport}
                className="text-xs font-semibold flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl transition-all shadow-sm bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                <Download size={15} /> <span>{t.downloadReport}</span>
              </button>
            )}
            <button
              onClick={resetUpload}
              className={`text-xs font-semibold flex items-center justify-center gap-1 px-3 py-2 rounded-xl transition-all border ${
                darkMode
                  ? "bg-gray-800 hover:bg-red-950 text-red-400 border-red-900/50"
                  : "bg-red-50 hover:bg-red-100 text-red-700 border-red-200"
              }`}
            >
              <Trash2 size={15} /> {t.clearButton}
            </button>
          </div>
        )}
      </div>

      {/* Target Language Selector */}
      <div className="mb-4">
        <label
          className={`block text-xs font-bold uppercase tracking-wider mb-2 ${
            darkMode ? "text-gray-400" : "text-gray-600"
          }`}
        >
          {t.selectLanguage}
        </label>
        <select
          value={language}
          onChange={(e) => setCustomLanguage(e.target.value)}
          className={`w-full p-2.5 sm:p-3 border-2 rounded-xl focus:outline-none transition-all text-sm font-medium shadow-sm ${
            darkMode
              ? "bg-gray-800 border-gray-700 text-gray-100 focus:border-green-400"
              : "bg-white border-gray-200 text-gray-900 focus:border-green-600"
          }`}
        >
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.flag} {lang.name}
            </option>
          ))}
        </select>
      </div>

      {/* Camera Capture Modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
          darkMode={darkMode}
          t={t}
        />
      )}

      {/* Mobile/Camera Action Bar */}
      <div className="mb-4 grid grid-cols-2 gap-3">
        <button
          onClick={() => setShowCamera(true)}
          className={`p-3.5 rounded-xl border transition-all duration-200 flex items-center justify-center gap-2 shadow-sm ${
            darkMode
              ? "bg-blue-600 hover:bg-blue-700 border-blue-500 text-white"
              : "bg-blue-600 hover:bg-blue-700 border-blue-500 text-white"
          }`}
        >
          <Camera size={18} />
          <span className="text-xs sm:text-sm font-semibold">{t.takePhoto}</span>
        </button>

        <button
          onClick={() => fileInputRef.current?.click()}
          className={`p-3.5 rounded-xl border transition-all duration-200 flex items-center justify-center gap-2 shadow-sm ${
            darkMode
              ? "bg-gray-800 hover:bg-gray-700 border-gray-700 text-emerald-400"
              : "bg-emerald-50 hover:bg-emerald-100 border-emerald-300 text-emerald-800"
          }`}
        >
          <ImageIcon size={18} />
          <span className="text-xs sm:text-sm font-semibold">{t.uploadFromGallery}</span>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileInput}
        />
      </div>

      {/* Drag & Drop Area */}
      {!preview && (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? darkMode
                ? "border-green-400 bg-green-950/30"
                : "border-green-500 bg-green-50"
              : darkMode
              ? "border-gray-700 hover:border-green-500 hover:bg-gray-800/50"
              : "border-gray-300 hover:border-green-500 hover:bg-emerald-50/50"
          }`}
        >
          <ImageIcon
            size={48}
            className={`mb-3 ${darkMode ? "text-gray-500" : "text-gray-400"}`}
          />
          <p className="text-sm font-semibold text-center">{t.uploadArea}</p>
          <p className={`text-xs mt-1 ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            {t.uploadSupports}
          </p>
        </div>
      )}

      {/* Image Preview */}
      {preview && (
        <div className="mt-4 flex flex-col items-center">
          <div className="relative group max-w-sm w-full">
            <img
              src={preview}
              alt="Leaf sample preview"
              decoding="async"
              onClick={() => setImageZoomed(!imageZoomed)}
              className={`w-full h-56 object-cover rounded-2xl shadow-lg border-2 cursor-pointer transition-all duration-300 ${
                darkMode ? "border-gray-700" : "border-emerald-200"
              } ${imageZoomed ? "scale-125 z-40" : "hover:brightness-95"}`}
            />
            <div
              className={`absolute top-2 right-2 px-2.5 py-1 rounded-full text-xs font-bold shadow-md ${
                darkMode ? "bg-emerald-700 text-white" : "bg-emerald-600 text-white"
              }`}
            >
              {t.readyToAnalyze}
            </div>
            <div
              className={`absolute bottom-2 left-2 px-2 py-0.5 rounded text-xs backdrop-blur-md ${
                darkMode ? "bg-gray-900/80 text-gray-200" : "bg-white/80 text-gray-800"
              }`}
            >
              🔍 {t.clickToZoom}
            </div>
          </div>
        </div>
      )}

      {/* Pre-Flight Image Quality Card */}
      {preview && (
        <div className="mt-4">
          {checkingQuality && (
            <div
              className={`p-3 rounded-xl border flex items-center gap-3 text-xs ${
                darkMode ? "bg-blue-950/40 border-blue-800 text-blue-300" : "bg-blue-50 border-blue-200 text-blue-800"
              }`}
            >
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent"></div>
              <span>{t.checkingQuality}</span>
            </div>
          )}

          {qualityCheck && !checkingQuality && (
            <div
              className={`p-3.5 rounded-xl border text-xs ${
                qualityCheck.quality_score >= 80
                  ? darkMode
                    ? "bg-emerald-950/30 border-emerald-800 text-emerald-300"
                    : "bg-emerald-50 border-emerald-200 text-emerald-900"
                  : qualityCheck.quality_score >= 60
                  ? darkMode
                    ? "bg-amber-950/30 border-amber-800 text-amber-300"
                    : "bg-amber-50 border-amber-200 text-amber-900"
                  : darkMode
                  ? "bg-red-950/30 border-red-800 text-red-300"
                  : "bg-red-50 border-red-200 text-red-900"
              }`}
            >
              <div className="flex items-center justify-between font-semibold">
                <span>
                  {qualityCheck.quality_score >= 80
                    ? `✅ ${t.qualityGood}`
                    : `⚠️ ${t.qualityWarning}`}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-white/10 font-bold">
                  {t.qualityScore}: {qualityCheck.quality_score}/100
                </span>
              </div>

              {qualityCheck.warnings && qualityCheck.warnings.length > 0 && (
                <ul className="mt-2 space-y-1 opacity-90">
                  {qualityCheck.warnings.map((w, idx) => (
                    <li key={idx}>• {w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}

      {/* Predict Button */}
      {preview && (
        <button
          onClick={predictDisease}
          disabled={loading || !file}
          className="mt-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white py-3.5 px-6 rounded-xl font-bold shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
              <span>{t.analyzing}</span>
            </>
          ) : (
            <>
              <Search size={18} />
              <span>{t.predictButton}</span>
            </>
          )}
        </button>
      )}

      {/* Error Message */}
      {error && (
        <div
          className={`mt-4 p-3.5 rounded-xl border flex items-start gap-2 text-xs sm:text-sm ${
            darkMode ? "bg-red-950/40 border-red-800 text-red-300" : "bg-red-50 border-red-200 text-red-700"
          }`}
        >
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-bold">{t.errorTitle}</p>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Predictions and Results Section */}
      {predictions.length > 0 && (
        <div ref={resultsRef} className="mt-8 space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-800 gap-2">
            <div className="flex items-center gap-2">
              <h3 className="text-lg sm:text-xl font-extrabold flex items-center gap-2">
                <Trophy size={20} className="text-amber-500" /> {t.predictionsTitle}
              </h3>
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold ${
                  darkMode ? "bg-emerald-900/60 text-emerald-300" : "bg-emerald-100 text-emerald-800"
                }`}
              >
                {predictions.length} {t.detected}
              </span>
            </div>

            {/* Ask AI CTA */}
            {onAskAi && (
              <button
                onClick={() => onAskAi(predictions, detectedDiseases)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 shadow-sm transition-all active:scale-95 ${
                  darkMode
                    ? "bg-gradient-to-r from-emerald-600 to-teal-700 text-white hover:from-emerald-500 hover:to-teal-600 border border-emerald-500/50"
                    : "bg-gradient-to-r from-emerald-600 to-teal-600 text-white hover:from-emerald-700 hover:to-teal-700 shadow-emerald-200"
                }`}
                title={t.askAboutResult || "Ask AI About This Diagnosis"}
              >
                <Bot size={15} />
                <span>{t.askAboutResult || "Ask AI About This Diagnosis"}</span>
              </button>
            )}
          </div>

          {/* Predictions Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            {predictions.map((pred, index) => {
              const severity = getSeverity(pred.confidence)
              return (
                <div
                  key={index}
                  className={`rounded-2xl p-4 transition-all duration-200 border ${
                    darkMode
                      ? "bg-gray-800/80 border-gray-700 hover:border-emerald-500"
                      : "bg-gradient-to-br from-white to-emerald-50/30 border-emerald-200 hover:shadow-md"
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {index === 0 ? (
                        <Medal size={22} className="text-amber-500 flex-shrink-0" />
                      ) : (
                        <Medal size={22} className="text-gray-400 flex-shrink-0" />
                      )}
                      <div>
                        <h4 className="font-bold text-sm leading-tight block">
                          {pred.label.replace(/___/g, " - ").replace(/_/g, " ")}
                        </h4>
                        <span className="text-xs text-gray-400">
                          {t.severity}: <strong className={severity.colorClass}>{severity.level}</strong>
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-center py-1">
                    <CircularProgress
                      percentage={pred.confidence * 100}
                      strokeColor={severity.stroke}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Multi-Disease Grad-CAM Localization Section */}
          {(visualizations.length > 0 || combinedVisualization) && (
            <div
              className={`rounded-2xl p-5 border shadow-md transition-all ${
                darkMode
                  ? "bg-gray-800/60 border-purple-900/50"
                  : "bg-gradient-to-br from-purple-50/50 to-indigo-50/30 border-purple-200"
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
                <div>
                  <h3 className="text-base sm:text-lg font-bold flex items-center gap-2">
                    🎯 {t.diseaseLocalization}
                  </h3>
                  <p className={`text-xs ${darkMode ? "text-gray-400" : "text-gray-600"}`}>
                    Visual attention overlays generated from DenseNet121 deep features
                  </p>
                </div>

                {/* View Switcher Toggle */}
                {combinedVisualization && visualizations.length > 0 && (
                  <div className="inline-flex rounded-xl p-1 bg-black/10 dark:bg-white/10 self-start sm:self-auto">
                    <button
                      onClick={() => setShowCombined(true)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                        showCombined
                          ? "bg-white dark:bg-gray-900 shadow-sm text-purple-700 dark:text-purple-300"
                          : "text-gray-500 hover:text-gray-800 dark:text-gray-400"
                      }`}
                    >
                      <Layers size={14} />
                      <span>{t.showCombined}</span>
                    </button>
                    <button
                      onClick={() => setShowCombined(false)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                        !showCombined
                          ? "bg-white dark:bg-gray-900 shadow-sm text-purple-700 dark:text-purple-300"
                          : "text-gray-500 hover:text-gray-800 dark:text-gray-400"
                      }`}
                    >
                      <LayoutGrid size={14} />
                      <span>{t.showIndividual}</span>
                    </button>
                  </div>
                )}
              </div>

              {/* Combined Multi-Disease View */}
              {showCombined && combinedVisualization && (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden border border-purple-300 dark:border-purple-800 bg-black/5 dark:bg-black/40">
                    <img
                      src={combinedVisualization}
                      alt="Combined multi-disease localization"
                      loading="lazy"
                      decoding="async"
                      className="w-full max-h-96 object-contain mx-auto"
                    />
                  </div>

                  {/* Multi-Disease Color Legend */}
                  <div className="flex flex-wrap items-center gap-2 pt-2">
                    <span className="text-xs font-bold text-gray-500 dark:text-gray-400">
                      Color Legend:
                    </span>
                    {visualizations.map((viz, idx) => (
                      <div
                        key={idx}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-white dark:bg-gray-800 border shadow-sm"
                        style={{ borderColor: viz.color }}
                      >
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: viz.color }}
                        ></span>
                        <span className="capitalize">{viz.disease.replace(/___/g, " ").replace(/_/g, " ")}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Individual Disease Breakdown Cards */}
              {(!showCombined || !combinedVisualization) && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {visualizations.map((viz, index) => (
                    <div
                      key={index}
                      className={`p-3.5 rounded-xl border-2 transition-all ${
                        darkMode ? "bg-gray-800" : "bg-white shadow-sm"
                      }`}
                      style={{ borderColor: viz.color }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className="w-3.5 h-3.5 rounded-full flex-shrink-0"
                          style={{ backgroundColor: viz.color }}
                        ></span>
                        <h4 className="font-bold text-xs sm:text-sm truncate">
                          {viz.disease.replace(/___/g, " - ").replace(/_/g, " ")}
                        </h4>
                      </div>

                      <img
                        src={viz.image}
                        alt={`${viz.disease.replace(/___/g, " - ").replace(/_/g, " ")} Grad-CAM`}
                        loading="lazy"
                        decoding="async"
                        className="w-full h-48 object-cover rounded-lg mb-2 border"
                      />

                      <div
                        className="text-xs font-bold py-1 px-2 rounded text-center capitalize"
                        style={{
                          backgroundColor: `${viz.color}20`,
                          color: viz.color,
                        }}
                      >
                        {t.highlighted} {viz.color}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Farmer Advisory Section */}
          {advice && (
            <div
              className={`rounded-2xl p-5 border shadow-md transition-all ${
                darkMode
                  ? "bg-gray-800/80 border-emerald-900/50"
                  : "bg-gradient-to-br from-emerald-50/50 to-teal-50/30 border-emerald-200"
              }`}
            >
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-3">
                <div className="flex items-center gap-2">
                  <Lightbulb size={22} className="text-amber-500" />
                  <h3 className="text-base sm:text-lg font-bold">{t.adviceTitle}</h3>
                </div>

                {/* Multilingual Translation Switcher & Ask AI CTA */}
                <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
                  {onAskAi && (
                    <button
                      onClick={() => onAskAi(predictions, detectedDiseases)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all shadow-sm active:scale-95 ${
                        darkMode
                          ? "bg-gray-700 hover:bg-emerald-800 text-emerald-300"
                          : "bg-emerald-600 hover:bg-emerald-700 text-white"
                      }`}
                      title={t.askAboutResult || "Ask AI Assistant"}
                    >
                      <Bot size={14} />
                      <span>{t.chatButton || "Ask AI Assistant"}</span>
                    </button>
                  )}

                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-gray-500 font-medium">
                      {t.translateTo}
                    </span>
                    <select
                      value={adviceLanguage}
                      onChange={(e) => translateAdvice(e.target.value)}
                      disabled={translating}
                      className={`p-1.5 border rounded-lg text-xs font-semibold focus:outline-none transition-all ${
                        darkMode
                          ? "bg-gray-800 border-gray-700 text-gray-100"
                          : "bg-white border-emerald-300 text-gray-900"
                      }`}
                    >
                      {languages.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.flag} {lang.name}
                        </option>
                      ))}
                    </select>
                    {translating && (
                      <div className="animate-spin rounded-full h-3.5 w-3.5 border-2 border-emerald-500 border-t-transparent"></div>
                    )}
                  </div>
                </div>
              </div>

              {translating ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-emerald-500 border-t-transparent"></div>
                  <span className="ml-2 text-xs font-semibold">{t.translating}</span>
                </div>
              ) : (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {renderAdvice(advice)}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
