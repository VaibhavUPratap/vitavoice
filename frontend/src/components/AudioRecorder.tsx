import React, { useState, useRef, useEffect } from 'react';
import {
  Mic, Square, Trash2, UploadCloud, Play, Pause,
  AlertCircle, ShieldCheck, CheckCircle2, Volume2,
} from 'lucide-react';

interface AudioRecorderProps {
  onUploadSuccess: (data: Record<string, unknown>) => void;
  onAnalysisStart: () => void;
  onAnalysisError: (err: string) => void;
  backendUrl: string;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onUploadSuccess, onAnalysisStart, onAnalysisError, backendUrl
}) => {
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [calibrationNoise, setCalibrationNoise] = useState<number | null>(null);
  const [noiseWarning, setNoiseWarning] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [clippingDetected, setClippingDetected] = useState(false);
  const [consentChecked, setConsentChecked] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const pcmBuffersRef = useRef<Float32Array[]>([]);
  const durationIntervalRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const micLevelMeterRef = useRef<HTMLDivElement | null>(null);
  const micLevelTextRef = useRef<HTMLSpanElement | null>(null);
  const frameCounterRef = useRef<number>(0);

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
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') audioContextRef.current.close();
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;
    analyserRef.current = null;
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
      const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      source.connect(processor); processor.connect(audioContext.destination);
      const rmsValues: number[] = [];
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        let ss = 0;
        for (let i = 0; i < inputData.length; i++) ss += inputData[i] * inputData[i];
        rmsValues.push(Math.sqrt(ss / inputData.length));
      };
      await new Promise((r) => setTimeout(r, 2000));
      processor.disconnect(); source.disconnect(); stream.getTracks().forEach((t) => t.stop()); audioContext.close();
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
      const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream); sourceRef.current = source;
      const analyser = audioContext.createAnalyser(); analyser.fftSize = 256; analyserRef.current = analyser;
      source.connect(analyser);
      const processor = audioContext.createScriptProcessor(4096, 1, 1); processorRef.current = processor;
      analyser.connect(processor); processor.connect(audioContext.destination);
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inputData.length; i++) {
          if (Math.abs(inputData[i]) >= 0.99) { setClippingDetected(true); break; }
        }
        pcmBuffersRef.current.push(new Float32Array(inputData));
      };
      setIsRecording(true); startVisualizer();
      let duration = 0;
      durationIntervalRef.current = window.setInterval(() => {
        duration += 1; setRecordingDuration(duration);
        if (duration >= 30) stopRecording();
      }, 1000);
    } catch (err) {
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
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const styles = getComputedStyle(document.documentElement);

    const draw = () => {
      const { width, height } = canvas;
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < bufferLength; i++) {
        const val = (dataArray[i] - 128) / 128;
        sumSquares += val * val;
      }
      const rms = Math.sqrt(sumSquares / bufferLength);
      const db = rms > 0 ? 20 * Math.log10(rms) : -100;
      const volumeLevel = Math.max(0, Math.min(100, ((db + 50) / 50) * 100));

      if (micLevelMeterRef.current) {
        micLevelMeterRef.current.style.width = `${volumeLevel}%`;
        micLevelMeterRef.current.style.background =
          volumeLevel > 85 ? 'var(--color-danger)' :
          volumeLevel > 60 ? 'var(--color-warning)' :
          'var(--color-accent)';
      }
      if (micLevelTextRef.current) micLevelTextRef.current.innerText = `${db.toFixed(1)} dB`;

      ctx.fillStyle = styles.getPropertyValue('--color-graphite').trim() || '#1a1f2e';
      ctx.fillRect(0, 0, width, height);

      frameCounterRef.current += 1;
      const t = frameCounterRef.current;
      const waves = [
        { amplitude: 50 * rms, frequency: 0.012, phase: t * 0.08, color: 'oklch(58% 0.20 256 / 0.85)', width: 2 },
        { amplitude: 30 * rms, frequency: 0.022, phase: -t * 0.12, color: 'oklch(58% 0.20 256 / 0.45)', width: 1.5 },
      ];
      waves.forEach((wave) => {
        ctx.beginPath();
        ctx.lineWidth = wave.width;
        ctx.strokeStyle = wave.color;
        for (let x = 0; x < width; x++) {
          const damping = Math.sin((x / width) * Math.PI);
          const y = height / 2 + damping * wave.amplitude * 120 * Math.sin(x * wave.frequency + wave.phase);
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    };
    draw();
  };

  const togglePlayback = () => {
    const player = audioPlayerRef.current;
    if (!player) return;
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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to establish server connection.';
      console.error(err);
      onAnalysisError(message);
      setErrorMsg(message);
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

  return (
    <article className="panel panel--recorder animate-fade-in">
      <header className="panel__header">
        <div>
          <h2 className="panel__title">Voice Screening Protocol</h2>
          <p className="panel__subtitle">Sustained vowel capture · 16 kHz mono WAV</p>
        </div>
        {isRecording && (
          <span className="badge badge--offline">
            <span className="badge__dot" />
            {recordingDuration}s / 30s
          </span>
        )}
      </header>

      <div className="recorder-layout">
        {/* Left Column: Calibration, Instructions, and Consent */}
        <div className="recorder-layout__col">
          {!calibrationDone && (
            <section className="step-section">
              <h3 className="step-section__title">
                <Volume2 style={{ width: 15, height: 15, color: 'var(--color-accent)' }} />
                Step 1: Acoustic Environment Calibration
              </h3>
              <p className="step-section__body" style={{ marginBottom: 'var(--space-md)' }}>
                Before recording, we measure background noise. Sit in a quiet room and click Calibrate — we listen for 2 seconds.
              </p>
              <button id="calibrate-btn" onClick={calibrateMicrophone} disabled={isCalibrating} className="btn btn--outline btn--sm">
                {isCalibrating ? 'Calibrating...' : 'Calibrate Environment'}
              </button>
            </section>
          )}

          {calibrationDone && !isRecording && !audioUrl && (
            <section className="step-section step-section--success">
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'flex-start' }}>
                  <CheckCircle2 style={{ width: 18, height: 18, color: 'var(--color-success)', flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <p className="step-section__title" style={{ marginBottom: 'var(--space-2xs)', color: 'var(--color-success)' }}>Environment Calibrated</p>
                    <p className="step-section__body">
                      Ambient noise: {(calibrationNoise || 0).toFixed(4)} RMS — suitable for biomarker extraction.
                    </p>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className="stat-block__label">Quality Score</span>
                  <p style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-success)', margin: '2px 0 0' }}>
                    {Math.round(100 - (calibrationNoise || 0) * 800)}%
                  </p>
                </div>
              </div>
            </section>
          )}

          {noiseWarning && !isRecording && (
            <section className="step-section step-section--warn">
              <p className="step-section__body" style={{ display: 'flex', gap: 'var(--space-2xs)', alignItems: 'flex-start' }}>
                <AlertCircle style={{ width: 16, height: 16, flexShrink: 0 }} />
                {noiseWarning}
              </p>
            </section>
          )}

          {clippingDetected && isRecording && (
            <section className="step-section step-section--error">
              <p className="step-section__body">
                <strong>Microphone clipping detected.</strong> Stand slightly further back to avoid digital distortion.
              </p>
            </section>
          )}

          <section className="step-section">
            <h3 className="step-section__title">Step 2: Sustained Vowel Vocalization</h3>
            <p className="step-section__body">
              Produce a sustained vowel <strong>&quot;ah&quot;</strong> sound for{' '}
              <strong>10–15 seconds</strong>. Hold the microphone about 6 inches from your mouth and maintain a steady pitch and volume.
            </p>
          </section>

          {!uploadProgress && (
            <section className="step-section">
              <h3 className="step-section__title">
                <ShieldCheck style={{ width: 15, height: 15, color: 'var(--color-accent)' }} />
                Clinical Consent Agreement
              </h3>
              <label className="consent-label">
                <input
                  id="consent-checkbox"
                  type="checkbox"
                  checked={consentChecked}
                  onChange={(e) => setConsentChecked(e.target.checked)}
                />
                <span>
                  I understand that VitaVoice is an AI-assisted vocal screening aid and not a definitive medical diagnosis.
                  I consent to upload my voice recording for vocal dysphonia biomarker screening.
                </span>
              </label>
            </section>
          )}
        </div>

        {/* Right Column: Visualizer Canvas / Upload and Action Buttons */}
        <div className="recorder-layout__col">
          <div className="visualizer" style={{ marginBottom: 'var(--space-md)' }}>
            <canvas ref={canvasRef} width={600} height={160} style={{ display: isRecording ? 'block' : 'none' }} />
            {!isRecording && !audioUrl && (
              <div
                className={`drop-zone${dragActive ? ' drop-zone--active' : ''}`}
                onDragEnter={handleDrag} onDragLeave={handleDrag}
                onDragOver={handleDrag} onDrop={handleDrop}
              >
                <UploadCloud style={{ width: 32, height: 32, color: 'var(--color-ink-3)', marginBottom: 8 }} />
                <p className="drop-zone__label">
                  Drag and drop audio here, or{' '}
                  <label className="drop-zone__browse">
                    browse files
                    <input type="file" className="sr-only" accept=".wav,.mp3" onChange={handleFileChange} />
                  </label>
                </p>
                <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)', margin: '6px 0 0' }}>Supports mono .wav or .mp3</p>
              </div>
            )}
            {!isRecording && audioUrl && (
              <div className="drop-zone" style={{ background: 'var(--color-paper-2)' }}>
                <p style={{ fontWeight: 600, color: 'var(--color-ink)', margin: '0 0 var(--space-sm)' }}>
                  Recording captured ({recordingDuration}s)
                </p>
                <audio ref={audioPlayerRef} src={audioUrl} onEnded={() => setIsPlaying(false)} className="sr-only" />
                <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                  <button id="playback-btn" onClick={togglePlayback} className="btn btn--ghost btn--sm">
                    {isPlaying ? <><Pause style={{ width: 14, height: 14 }} /> Pause</> : <><Play style={{ width: 14, height: 14 }} /> Play Preview</>}
                  </button>
                  <button id="delete-recording-btn" onClick={deleteRecording} className="btn btn--danger btn--sm">
                    <Trash2 style={{ width: 14, height: 14 }} /> Delete
                  </button>
                </div>
              </div>
            )}
          </div>

          {isRecording && (
            <div className="meter" style={{ marginBottom: 'var(--space-md)' }}>
              <div className="meter__row">
                <span>Real-time microphone level</span>
                <span ref={micLevelTextRef} style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-accent)' }}>-50.0 dB</span>
              </div>
              <div className="meter__track">
                <div ref={micLevelMeterRef} className="meter__fill" style={{ width: '0%' }} />
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
            {!audioBlob ? (
              <button
                id="record-btn"
                onClick={isRecording ? stopRecording : startRecording}
                disabled={!calibrationDone}
                className={`btn btn--primary btn--lg${isRecording ? ' btn--danger' : ''}`}
                style={{ width: '100%' }}
              >
                {isRecording ? (
                  <><Square style={{ width: 16, height: 16 }} /> Stop Recording (Min 10s)</>
                ) : (
                  <><Mic style={{ width: 16, height: 16 }} /> Begin Voice Recording</>
                )}
              </button>
            ) : (
              <button
                id="submit-screening-btn"
                onClick={() => uploadAudio(audioBlob)}
                disabled={uploadProgress || !consentChecked}
                className="btn btn--primary btn--lg"
                style={{ width: '100%' }}
              >
                {uploadProgress ? 'Processing Voice Biomarkers...' : 'Execute AI Risk Screening'}
              </button>
            )}
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="step-section step-section--error" style={{ marginTop: 'var(--space-md)' }}>
          <p className="step-section__body" style={{ display: 'flex', gap: 'var(--space-2xs)' }}>
            <AlertCircle style={{ width: 18, height: 18, flexShrink: 0 }} />
            {errorMsg}
          </p>
        </div>
      )}
    </article>
  );
};
