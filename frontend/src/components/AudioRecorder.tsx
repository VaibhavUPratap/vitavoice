import React, { useState, useRef, useEffect } from 'react';
import {
  Mic, Square, Trash2, UploadCloud, Play, Pause,
  AlertCircle, ShieldCheck, CheckCircle2, Volume2,
} from 'lucide-react';

interface AudioRecorderProps {
  onUploadSuccess: (data: any) => void;
  onAnalysisStart: () => void;
  onAnalysisError: (err: string) => void;
  backendUrl: string;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onUploadSuccess, onAnalysisStart, onAnalysisError, backendUrl
}) => {
  const [isCalibrating, setIsCalibrating]     = useState(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [calibrationNoise, setCalibrationNoise] = useState<number | null>(null);
  const [noiseWarning, setNoiseWarning]         = useState<string | null>(null);
  const [isRecording, setIsRecording]           = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioUrl, setAudioUrl]   = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [clippingDetected, setClippingDetected] = useState(false);
  const [consentChecked, setConsentChecked]     = useState(false);
  const [dragActive, setDragActive]             = useState(false);
  const [uploadProgress, setUploadProgress]     = useState(false);
  const [errorMsg, setErrorMsg]                 = useState<string | null>(null);

  const audioContextRef  = useRef<AudioContext | null>(null);
  const streamRef        = useRef<MediaStream | null>(null);
  const processorRef     = useRef<ScriptProcessorNode | null>(null);
  const sourceRef        = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef      = useRef<AnalyserNode | null>(null);
  const pcmBuffersRef    = useRef<Float32Array[]>([]);
  const durationIntervalRef  = useRef<number | null>(null);
  const audioPlayerRef       = useRef<HTMLAudioElement | null>(null);
  const canvasRef            = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef    = useRef<number | null>(null);
  const micLevelMeterRef     = useRef<HTMLDivElement | null>(null);
  const micLevelTextRef      = useRef<HTMLSpanElement | null>(null);
  const frameCounterRef      = useRef<number>(0);

  useEffect(() => {
    return () => {
      cleanupAudioNodes();
      if (durationIntervalRef.current) clearInterval(durationIntervalRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  const cleanupAudioNodes = () => {
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current.onaudioprocess = null; }
    if (sourceRef.current) sourceRef.current.disconnect();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') audioContextRef.current.close();
    processorRef.current = null; sourceRef.current = null;
    streamRef.current = null; audioContextRef.current = null; analyserRef.current = null;
  };

  const writeWavHeader = (view: DataView, offset: number, sampleRate: number, numChannels: number, bitsPerSample: number, dataLength: number) => {
    const ws = (v: DataView, o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ws(view, offset, 'RIFF'); view.setUint32(offset + 4, 36 + dataLength, true); ws(view, offset + 8, 'WAVE');
    ws(view, offset + 12, 'fmt '); view.setUint32(offset + 16, 16, true); view.setUint16(offset + 20, 1, true);
    view.setUint16(offset + 22, numChannels, true); view.setUint32(offset + 24, sampleRate, true);
    view.setUint32(offset + 28, sampleRate * numChannels * (bitsPerSample / 8), true);
    view.setUint16(offset + 32, numChannels * (bitsPerSample / 8), true); view.setUint16(offset + 34, bitsPerSample, true);
    ws(view, offset + 36, 'data'); view.setUint32(offset + 40, dataLength, true);
  };

  const encodeWav = (samples: Float32Array, sampleRate: number): Blob => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeWavHeader(view, 0, sampleRate, 1, 16, samples.length * 2);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  };

  const calibrateMicrophone = async () => {
    try {
      setErrorMsg(null); setNoiseWarning(null); setIsCalibrating(true);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      source.connect(processor); processor.connect(audioContext.destination);
      const rmsValues: number[] = [];
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        let ss = 0; for (let i = 0; i < inputData.length; i++) ss += inputData[i] * inputData[i];
        rmsValues.push(Math.sqrt(ss / inputData.length));
      };
      await new Promise(r => setTimeout(r, 2000));
      processor.disconnect(); source.disconnect(); stream.getTracks().forEach(t => t.stop()); audioContext.close();
      const avgRMS = rmsValues.length > 0 ? rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length : 0;
      setCalibrationNoise(avgRMS); setCalibrationDone(true); setIsCalibrating(false);
      if (avgRMS > 0.05) setNoiseWarning('High ambient noise detected. Please move to a quieter environment before recording.');
      else setNoiseWarning(null);
    } catch (err) {
      console.error(err); setIsCalibrating(false);
      setErrorMsg('Failed to access microphone for calibration. Please check permissions.');
    }
  };

  const startRecording = async () => {
    try {
      setErrorMsg(null); setAudioUrl(null); setAudioBlob(null);
      setClippingDetected(false); pcmBuffersRef.current = []; setRecordingDuration(0);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
      streamRef.current = stream;
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream); sourceRef.current = source;
      const analyser = audioContext.createAnalyser(); analyser.fftSize = 256; analyserRef.current = analyser;
      source.connect(analyser);
      const processor = audioContext.createScriptProcessor(4096, 1, 1); processorRef.current = processor;
      analyser.connect(processor); processor.connect(audioContext.destination);
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inputData.length; i++) { if (Math.abs(inputData[i]) >= 0.99) { setClippingDetected(true); break; } }
        pcmBuffersRef.current.push(new Float32Array(inputData));
      };
      setIsRecording(true); startVisualizer();
      let duration = 0;
      durationIntervalRef.current = window.setInterval(() => {
        duration += 1; setRecordingDuration(duration);
        if (duration >= 30) stopRecording();
      }, 1000);
    } catch (err: any) {
      console.error(err); setErrorMsg('Microphone access denied. Please check permissions.');
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (durationIntervalRef.current) { clearInterval(durationIntervalRef.current); durationIntervalRef.current = null; }
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    const totalLength = pcmBuffersRef.current.reduce((acc, buf) => acc + buf.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of pcmBuffersRef.current) { result.set(buf, offset); offset += buf.length; }
    if (totalLength < 16000 * 10) {
      setErrorMsg('Voice sample is too short. Please record for at least 10 seconds.');
      cleanupAudioNodes(); return;
    }
    const sampleRate = audioContextRef.current?.sampleRate || 16000;
    const wavBlob = encodeWav(result, sampleRate);
    setAudioBlob(wavBlob); setAudioUrl(URL.createObjectURL(wavBlob));
    cleanupAudioNodes();
  };

  const startVisualizer = () => {
    const canvas = canvasRef.current; const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const draw = () => {
      const { width, height } = canvas;
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < bufferLength; i++) { const val = (dataArray[i] - 128) / 128; sumSquares += val * val; }
      const rms = Math.sqrt(sumSquares / bufferLength);
      const db = rms > 0 ? 20 * Math.log10(rms) : -100;
      const volumeLevel = Math.max(0, Math.min(100, ((db + 50) / 50) * 100));
      if (micLevelMeterRef.current) {
        micLevelMeterRef.current.style.width = `${volumeLevel}%`;
        micLevelMeterRef.current.style.background =
          volumeLevel > 85 ? 'oklch(62% 0.22 25)' :
          volumeLevel > 60 ? 'oklch(74% 0.18 60)' :
          'var(--color-accent)';
      }
      if (micLevelTextRef.current) micLevelTextRef.current.innerText = `${db.toFixed(1)} dB`;

      ctx.fillStyle = 'oklch(10% 0.025 275)';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = 'oklch(20% 0.028 270 / 0.3)';
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 40) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke(); }
      for (let i = 0; i < height; i += 20) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke(); }

      frameCounterRef.current += 1;
      const t = frameCounterRef.current;
      const waves = [
        { amplitude: 50 * rms, frequency: 0.012, phase: t * 0.08, color: 'oklch(72% 0.20 215 / 0.85)', width: 2.5 },
        { amplitude: 30 * rms, frequency: 0.022, phase: -t * 0.12, color: 'oklch(68% 0.22 290 / 0.55)', width: 1.5 },
        { amplitude: 15 * rms, frequency: 0.008, phase: t * 0.05, color: 'oklch(60% 0.18 240 / 0.35)', width: 1.0 },
      ];
      waves.forEach(wave => {
        ctx.beginPath(); ctx.lineWidth = wave.width; ctx.strokeStyle = wave.color;
        if (wave.amplitude > 0.05 && wave.width > 2) { ctx.shadowBlur = 14; ctx.shadowColor = wave.color; }
        else ctx.shadowBlur = 0;
        for (let x = 0; x < width; x++) {
          const damping = Math.sin((x / width) * Math.PI);
          const y = height / 2 + damping * wave.amplitude * 120 * Math.sin(x * wave.frequency + wave.phase);
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      ctx.shadowBlur = 0;
    };
    draw();
  };

  const togglePlayback = () => {
    const player = audioPlayerRef.current; if (!player) return;
    if (isPlaying) { player.pause(); setIsPlaying(false); }
    else { player.play(); setIsPlaying(true); }
  };

  const deleteRecording = () => {
    setAudioUrl(null); setAudioBlob(null); setIsPlaying(false);
    setErrorMsg(null); setConsentChecked(false); setClippingDetected(false);
  };

  const getQualityScore = () => {
    const noise = calibrationNoise !== null ? calibrationNoise : 0.002;
    let score = 100 - noise * 800;
    score = Math.max(30, Math.min(100, score));
    if (clippingDetected) score -= 20;
    return Math.round(score);
  };

  const uploadAudio = async (blobToUpload: Blob) => {
    if (!consentChecked) { setErrorMsg('You must accept the clinical screening disclaimer before proceeding.'); return; }
    onAnalysisStart(); setUploadProgress(true); setErrorMsg(null);
    const formData = new FormData();
    formData.append('file', blobToUpload, 'voice_sample.wav');
    try {
      const response = await fetch(`${backendUrl}/api/v1/screen`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Server failed to process voice sample.');
      onUploadSuccess({ ...data, recording_quality_score: getQualityScore() });
    } catch (err: any) {
      console.error(err);
      onAnalysisError(err.message || 'Failed to establish server connection.');
      setErrorMsg(err.message || 'Failed to communicate with inference engine.');
    } finally {
      setUploadProgress(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0];
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (ext !== 'wav' && ext !== 'mp3') { setErrorMsg('Please upload a standard WAV or MP3 audio file.'); return; }
      setCalibrationDone(true); uploadAudio(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) { setCalibrationDone(true); uploadAudio(e.target.files[0]); }
  };

  /* ── Shared style helpers ── */
  const sectionStyle: React.CSSProperties = {
    background: 'oklch(14% 0.028 270 / 0.55)',
    border: '1px solid var(--color-rule)',
    borderRadius: 12,
    padding: '1.25rem 1.5rem',
    marginBottom: '1.25rem',
    backdropFilter: 'blur(12px)',
  };

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div
        className="glass"
        style={{
          width: '100%',
          maxWidth: 640,
          padding: '2rem',
          marginBottom: '1.5rem',
          boxShadow: '0 0 60px -16px var(--color-glow-cyan)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.75rem' }}>
          <h2
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.1875rem',
              fontWeight: 800,
              color: 'var(--color-accent)',
            }}
          >
            Voice Screening Protocol
          </h2>
          {isRecording && (
            <span
              style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                color: 'var(--color-danger)',
                fontFamily: 'var(--font-display)',
                fontSize: '0.875rem',
                fontWeight: 600,
                animation: 'fadeInUp 0.3s ease forwards',
              }}
            >
              <span
                style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: 'var(--color-danger)',
                  display: 'inline-block',
                  animation: 'recordingPulse 1.5s ease-in-out infinite',
                }}
              />
              {recordingDuration}s / 30s
            </span>
          )}
        </div>

        {/* STEP 1 — Calibration */}
        {!calibrationDone && (
          <div style={sectionStyle}>
            <h3
              style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                fontFamily: 'var(--font-display)',
                fontSize: '0.8125rem',
                fontWeight: 700,
                color: 'var(--color-ink)',
                marginBottom: '0.75rem',
              }}
            >
              <Volume2 style={{ width: 15, height: 15, color: 'var(--color-accent)', flexShrink: 0 }} />
              Step 1: Acoustic Environment Calibration
            </h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--color-ink-2)', lineHeight: 1.7, marginBottom: '1.25rem' }}>
              Before recording, we measure background noise. Sit in a quiet room and click Calibrate — we listen for 2 seconds.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
              <button
                id="calibrate-btn"
                onClick={calibrateMicrophone}
                disabled={isCalibrating}
                className="btn-ghost btn-sm"
              >
                {isCalibrating ? 'Calibrating...' : 'Calibrate Environment'}
              </button>
              {isCalibrating && (
                <span
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--color-accent)',
                    animation: 'fadeInUp 0.3s ease forwards',
                  }}
                >
                  Analyzing ambient noise... stay silent.
                </span>
              )}
            </div>
          </div>
        )}

        {/* STEP 2 — Calibration success */}
        {calibrationDone && !isRecording && !audioUrl && (
          <div
            style={{
              ...sectionStyle,
              background: 'oklch(16% 0.06 155 / 0.30)',
              border: '1px solid oklch(40% 0.10 155 / 0.30)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                <CheckCircle2 style={{ width: 18, height: 18, color: 'var(--color-success)', flexShrink: 0, marginTop: 1 }} />
                <div>
                  <p style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-success)', marginBottom: '0.25rem' }}>
                    Environment Calibrated
                  </p>
                  <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-2)', lineHeight: 1.6 }}>
                    Ambient noise: {(calibrationNoise || 0).toFixed(4)} RMS — suitable for biomarker extraction.
                  </p>
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0, borderLeft: '1px solid oklch(40% 0.10 155 / 0.30)', paddingLeft: '1rem' }}>
                <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block' }}>Quality Score</span>
                <span style={{ fontSize: '1.5rem', fontWeight: 900, fontFamily: 'var(--font-display)', color: 'var(--color-success)', display: 'block', lineHeight: 1.2, marginTop: 2 }}>
                  {Math.round(100 - (calibrationNoise || 0) * 800)}%
                </span>
                <span style={{ fontSize: '0.5625rem', color: 'var(--color-success)', fontWeight: 700, textTransform: 'uppercase' }}>
                  {(calibrationNoise || 0) < 0.01 ? 'Excellent' : (calibrationNoise || 0) < 0.05 ? 'Good' : 'Noisy'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Noise warning */}
        {noiseWarning && !isRecording && (
          <div
            style={{
              ...sectionStyle,
              background: 'oklch(18% 0.06 60 / 0.40)',
              border: '1px solid oklch(40% 0.10 60 / 0.30)',
              color: 'var(--color-warning)',
              display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
            }}
          >
            <AlertCircle style={{ width: 16, height: 16, flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: '0.75rem', lineHeight: 1.6 }}>{noiseWarning}</span>
          </div>
        )}

        {/* Clipping warning */}
        {clippingDetected && isRecording && (
          <div
            style={{
              ...sectionStyle,
              background: 'oklch(18% 0.06 25 / 0.50)',
              border: '1px solid oklch(40% 0.10 25 / 0.40)',
              color: 'var(--color-danger)',
              display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
              animation: 'fadeInUp 0.3s ease',
            }}
          >
            <AlertCircle style={{ width: 16, height: 16, flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: '0.75rem', lineHeight: 1.6 }}>
              <strong>Microphone Clipping Detected:</strong> Your voice is too loud. Stand slightly further back to avoid digital distortion.
            </span>
          </div>
        )}

        {/* Recording guidelines */}
        <p style={{ fontSize: '0.875rem', color: 'var(--color-ink-2)', lineHeight: 1.7, marginBottom: '1.25rem' }}>
          Produce a sustained vowel <strong style={{ color: 'var(--color-ink)' }}>"ah"</strong> sound (constant pitch and volume) for{' '}
          <strong style={{ color: 'var(--color-ink)' }}>10–15 seconds</strong>. Hold the microphone about 6 inches from your mouth.
        </p>

        {/* Visualizer / Drop zone */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: 160,
            background: 'oklch(9% 0.022 272)',
            border: `1px solid ${dragActive ? 'var(--color-accent)' : 'var(--color-rule)'}`,
            borderRadius: 12,
            overflow: 'hidden',
            marginBottom: '1.25rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'border-color var(--dur-fast) var(--ease-out)',
          }}
        >
          <canvas
            ref={canvasRef}
            width={600}
            height={160}
            style={{ width: '100%', height: '100%', display: isRecording ? 'block' : 'none' }}
          />

          {!isRecording && !audioUrl && (
            <div
              style={{
                position: 'absolute', inset: 0,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                borderRadius: 12,
                border: `2px dashed ${dragActive ? 'var(--color-accent)' : 'var(--color-rule)'}`,
                background: dragActive ? 'oklch(20% 0.08 215 / 0.15)' : 'transparent',
                transition: 'all var(--dur-fast) var(--ease-out)',
              }}
              onDragEnter={handleDrag} onDragLeave={handleDrag}
              onDragOver={handleDrag} onDrop={handleDrop}
            >
              <UploadCloud style={{ width: 36, height: 36, color: 'var(--color-ink-3)', marginBottom: 10 }} />
              <p style={{ fontSize: '0.875rem', color: 'var(--color-ink-2)', fontWeight: 500, textAlign: 'center' }}>
                Drag and drop audio here, or{' '}
                <label
                  style={{
                    color: 'var(--color-accent)',
                    textDecoration: 'underline',
                    cursor: 'pointer',
                    fontWeight: 600,
                  }}
                >
                  browse files
                  <input type="file" style={{ display: 'none' }} accept=".wav,.mp3" onChange={handleFileChange} />
                </label>
              </p>
              <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', marginTop: 6 }}>Supports mono .wav or .mp3</p>
            </div>
          )}

          {!isRecording && audioUrl && (
            <div
              style={{
                position: 'absolute', inset: 0,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                background: 'oklch(12% 0.025 270 / 0.80)',
                gap: '1rem',
              }}
            >
              <p style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-ink)', textAlign: 'center' }}>
                Recording captured ({recordingDuration}s) ✓
              </p>
              <audio ref={audioPlayerRef} src={audioUrl} onEnded={() => setIsPlaying(false)} style={{ display: 'none' }} />
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button
                  id="playback-btn"
                  onClick={togglePlayback}
                  className="btn-ghost btn-sm"
                >
                  {isPlaying
                    ? <><Pause style={{ width: 14, height: 14 }} /> Pause</>
                    : <><Play style={{ width: 14, height: 14 }} /> Play Preview</>
                  }
                </button>
                <button id="delete-recording-btn" onClick={deleteRecording} className="btn-danger btn-sm">
                  <Trash2 style={{ width: 14, height: 14 }} /> Delete
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Mic level meter */}
        {isRecording && (
          <div style={{ ...sectionStyle, marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.625rem' }}>
              <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--color-ink-2)', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--color-accent)', display: 'inline-block', animation: 'recordingPulse 2s ease-in-out infinite' }} />
                Real-time Microphone Level
              </span>
              <span ref={micLevelTextRef} style={{ fontSize: '0.6875rem', fontFamily: 'var(--font-mono)', color: 'var(--color-accent)' }}>
                -50.0 dB
              </span>
            </div>
            <div style={{ width: '100%', height: 6, background: 'oklch(14% 0.022 270)', borderRadius: 3, overflow: 'hidden' }}>
              <div ref={micLevelMeterRef} className="mic-meter-fill" style={{ width: '0%' }} />
            </div>
          </div>
        )}

        {/* Consent */}
        {!uploadProgress && (
          <div style={sectionStyle}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-display)', fontSize: '0.8125rem', fontWeight: 700, color: 'var(--color-ink)', marginBottom: '0.875rem' }}>
              <ShieldCheck style={{ width: 15, height: 15, color: 'var(--color-accent)', flexShrink: 0 }} />
              Clinical Consent Agreement
            </h3>
            <label
              style={{
                display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
                cursor: 'pointer',
                color: 'var(--color-ink-2)',
              }}
            >
              <input
                id="consent-checkbox"
                type="checkbox"
                checked={consentChecked}
                onChange={e => setConsentChecked(e.target.checked)}
                style={{ marginTop: 3, width: 16, height: 16, accentColor: 'var(--color-accent)', flexShrink: 0 }}
              />
              <span style={{ fontSize: '0.6875rem', lineHeight: 1.7 }}>
                I understand that VitaVoice is an AI-assisted vocal screening aid and not a definitive medical diagnosis. I consent to upload my voice recording for vocal dysphonia biomarker screening, and I understand this should be reviewed with an Otolaryngologist or Neurologist.
              </span>
            </label>
          </div>
        )}

        {/* Action controls */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'center' }}>
          {!audioBlob ? (
            <button
              id="record-btn"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!calibrationDone}
              className={isRecording ? 'btn-primary' : 'btn-primary glow-cyan'}
              style={{
                minWidth: 220,
                background: isRecording
                  ? 'linear-gradient(135deg, oklch(52% 0.22 25), oklch(48% 0.20 25))'
                  : undefined,
                animation: isRecording ? 'recordingPulse 1.5s ease-in-out infinite' : undefined,
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', zIndex: 1 }}>
                {isRecording
                  ? <><Square style={{ width: 16, height: 16 }} /> Stop Recording (Min 10s)</>
                  : <><Mic style={{ width: 16, height: 16 }} /> Begin Voice Recording</>
                }
              </span>
            </button>
          ) : (
            <button
              id="submit-screening-btn"
              onClick={() => uploadAudio(audioBlob)}
              disabled={uploadProgress || !consentChecked}
              className="btn-primary glow-green"
              style={{
                minWidth: 220,
                background: 'linear-gradient(135deg, oklch(60% 0.18 155), oklch(56% 0.16 175))',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', zIndex: 1 }}>
                {uploadProgress ? (
                  <>
                    <svg style={{ animation: 'ringRotate 1s linear infinite', width: 16, height: 16 }} viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" />
                      <path fill="currentColor" opacity="0.75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Processing Voice Biomarkers...
                  </>
                ) : (
                  'Execute AI Risk Screening'
                )}
              </span>
            </button>
          )}
        </div>

        {/* Error */}
        {errorMsg && (
          <div
            style={{
              marginTop: '1.25rem',
              padding: '1rem 1.25rem',
              background: 'oklch(18% 0.06 25 / 0.50)',
              border: '1px solid oklch(40% 0.10 25 / 0.40)',
              borderRadius: 12,
              color: 'var(--color-danger)',
              fontSize: '0.875rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.75rem',
            }}
          >
            <AlertCircle style={{ width: 18, height: 18, flexShrink: 0, marginTop: 1 }} />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>
    </div>
  );
};
