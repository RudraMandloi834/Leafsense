import { useState, useRef, useEffect, useCallback } from 'react'
import { X, Camera, RotateCw, Check, AlertCircle } from 'lucide-react'

export default function CameraCapture({ onCapture, onClose, darkMode, t }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const [stream, setStream] = useState(null)
    const [facingMode, setFacingMode] = useState('environment') // 'user' for front, 'environment' for back
    const [error, setError] = useState('')
    const [capturedImage, setCapturedImage] = useState(null)

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop())
            setStream(null)
        }
    }, [stream])

    const startCamera = useCallback(async () => {
        try {
            setError('')

            if (stream) {
                stream.getTracks().forEach(track => track.stop())
            }

            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: facingMode,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            })

            setStream(mediaStream)

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream
            }
        } catch (err) {
            console.error('Camera error:', err)
            setError(t?.cameraError || 'Camera access denied or unavailable')
        }
    }, [facingMode, stream, t])

    useEffect(() => {
        let activeStream = null
        let isCancelled = false

        const initCamera = async () => {
            try {
                const mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 1920 },
                        height: { ideal: 1080 }
                    },
                    audio: false
                })
                if (isCancelled) {
                    mediaStream.getTracks().forEach(track => track.stop())
                    return
                }
                activeStream = mediaStream
                setStream(mediaStream)
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream
                }
            } catch (err) {
                if (!isCancelled) {
                    console.error('Camera error:', err)
                    setError(t?.cameraError || 'Camera access denied or unavailable')
                }
            }
        }

        initCamera()

        return () => {
            isCancelled = true
            if (activeStream) {
                activeStream.getTracks().forEach(track => track.stop())
            }
        }
    }, [facingMode, t])

    const capturePhoto = () => {
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current

        // Set canvas dimensions to match video
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        // Draw video frame to canvas
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Convert to blob
        canvas.toBlob((blob) => {
            if (blob) {
                const imageUrl = URL.createObjectURL(blob)
                setCapturedImage(imageUrl)
                stopCamera()
            }
        }, 'image/jpeg', 0.95)
    }

    const retake = () => {
        setCapturedImage(null)
        startCamera()
    }

    const usePhoto = () => {
        if (!capturedImage || !canvasRef.current) return

        canvasRef.current.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], 'leaf-camera-capture.jpg', { type: 'image/jpeg' })
                onCapture(file, capturedImage)
                stopCamera()
                onClose()
            }
        }, 'image/jpeg', 0.95)
    }

    const switchCamera = () => {
        setFacingMode(prev => prev === 'user' ? 'environment' : 'user')
    }

    return (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
            {/* Header */}
            <div className={`p-4 flex justify-between items-center ${darkMode ? 'bg-gray-900' : 'bg-gray-800'}`}>
                <button
                    onClick={() => {
                        stopCamera()
                        onClose()
                    }}
                    className="text-white text-2xl p-2 hover:bg-gray-700 rounded-lg transition-colors"
                    aria-label="Close camera"
                >
                    <X size={24} />
                </button>
                <h2 className="text-white font-semibold">{t?.takePhoto || "Take Photo"}</h2>
                <div className="w-10"></div>
            </div>

            {/* Camera View or Captured Image */}
            <div className="flex-1 relative flex items-center justify-center bg-black">
                {error ? (
                    <div className="text-center p-6">
                        <div className="flex items-center justify-center gap-2 text-red-400 mb-4">
                            <AlertCircle size={24} />
                            <p>{error}</p>
                        </div>
                        <button
                            onClick={startCamera}
                            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2 mx-auto"
                        >
                            <RotateCw size={20} />
                            Try Again
                        </button>
                    </div>
                ) : capturedImage ? (
                    <img
                        src={capturedImage}
                        alt="Captured"
                        className="max-w-full max-h-full object-contain"
                    />
                ) : (
                    <>
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="max-w-full max-h-full object-contain"
                        />
                        <canvas ref={canvasRef} className="hidden" />
                    </>
                )}
            </div>

            {/* Controls */}
            <div className={`p-6 ${darkMode ? 'bg-gray-900' : 'bg-gray-800'}`}>
                {capturedImage ? (
                    <div className="flex gap-4 justify-center">
                        <button
                            onClick={retake}
                            className="flex-1 max-w-xs bg-gray-600 hover:bg-gray-700 text-white py-4 rounded-xl font-semibold transition-colors flex items-center justify-center gap-2"
                        >
                            <RotateCw size={20} /> {t?.retakePhoto || "Retake Photo"}
                        </button>
                        <button
                            onClick={usePhoto}
                            className="flex-1 max-w-xs bg-green-600 hover:bg-green-700 text-white py-4 rounded-xl font-semibold transition-colors flex items-center justify-center gap-2"
                        >
                            <Check size={20} /> {t?.usePhoto || "Use Photo"}
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-col gap-4">
                        <div className="flex gap-4 justify-center items-center">
                            <button
                                onClick={switchCamera}
                                className="bg-gray-700 hover:bg-gray-600 text-white p-4 rounded-full transition-colors"
                                title={t?.switchCamera || "Switch Camera"}
                            >
                                <RotateCw size={24} />
                            </button>

                            <button
                                onClick={capturePhoto}
                                disabled={!stream}
                                className="w-20 h-20 bg-white hover:bg-gray-200 rounded-full border-4 border-gray-300 transition-all disabled:opacity-50 flex items-center justify-center"
                                aria-label="Capture photograph"
                            >
                                <Camera size={32} className="text-gray-700" />
                            </button>

                            <div className="w-14"></div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
