import React, { useState, useEffect } from 'react';
import {
  ShieldCheck, Activity, Award, ServerCrash, Cpu,
  RefreshCw, Sun, Moon, Mic, Zap, Brain
} from 'lucide-react';
import { AudioRecorder } from './components/AudioRecorder';
import { Dashboard } from './components/Dashboard';

const BACKEND_URL = 'http://localhost:8000';

/* ─── Floating Orb (3D atmospheric background element) ─── */
function HeroOrb({
  size, color, top, left, blur, opacity, animClass
}: {
  size: number; color: string; top: string; left: string;
  blur: number; opacity: number; animClass: string;
}) {
  return (
    <div
      aria-hidden="true"
      className={`hero-orb ${animClass}`}
      style={{
        width: size, height: size,
        background: color,
        top, left,
        filter: `blur(${blur}px)`,
        opacity,
        zIndex: 0,
      }}
    />
  );
}

/* ─── Stat Tile ─── */
function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-tile flex-1 min-w-[160px]">
      <p
        style={{
          fontSize: '0.625rem',
          color: 'var(--color-ink-3)',
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.10em',
          marginBottom: '0.375rem',
        }}
      >
        {label}
      </p>
      <p
        style={{
          fontSize: '1.0625rem',
          fontWeight: 800,
          fontFamily: 'var(--font-display)',
          color: 'var(--color-ink)',
        }}
      >
        {value}
      </p>
    </div>
  );
}

/* ─── Processing Loader ─── */
function AnalyzingScreen({ message }: { message: string }) {
  return (
    <div
      className="glass animate-fade-in-scale"
      style={{
        maxWidth: 460,
        width: '100%',
        padding: '3rem 2.5rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 0 60px -12px var(--color-glow-cyan)',
      }}
    >
      {/* Laser scanline */}
      <div
        className="scanline"
        style={{ position: 'absolute', left: 0, right: 0, height: 40 }}
      />

      {/* Animated ring */}
      <div style={{ position: 'relative', marginBottom: '2rem' }}>
        <svg
          width={80} height={80}
          viewBox="0 0 80 80"
          style={{ animation: 'ringRotate 2s linear infinite' }}
        >
          <circle cx="40" cy="40" r="35"
            stroke="oklch(72% 0.20 215 / 0.15)"
            strokeWidth="3" fill="none"
          />
          <circle cx="40" cy="40" r="35"
            stroke="url(#ringGrad)"
            strokeWidth="3" fill="none"
            strokeLinecap="round"
            strokeDasharray="60 160"
          />
          <defs>
            <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="var(--color-accent)" />
              <stop offset="100%" stopColor="var(--color-accent-warm)" />
            </linearGradient>
          </defs>
        </svg>
        <Activity
          style={{
            position: 'absolute', inset: 0,
            margin: 'auto', width: 28, height: 28,
            color: 'var(--color-accent)',
          }}
        />
      </div>

      <h3
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.25rem',
          fontWeight: 800,
          color: 'var(--color-ink)',
          marginBottom: '0.5rem',
        }}
      >
        Analyzing Voice Biomarkers
      </h3>
      <p
        style={{
          fontSize: '0.6875rem',
          color: 'var(--color-ink-3)',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          marginBottom: '2rem',
        }}
      >
        Running Digital Signal Inference
      </p>

      {/* Progress bar */}
      <div
        style={{
          width: '100%', height: 3,
          background: 'oklch(22% 0.028 270)',
          borderRadius: 9999,
          overflow: 'hidden',
          marginBottom: '1.5rem',
        }}
      >
        <div
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, var(--color-accent), var(--color-accent-warm))',
            borderRadius: 9999,
            animation: 'progress 16s ease-out forwards',
          }}
        />
      </div>

      <p
        style={{
          fontSize: '0.875rem',
          color: 'var(--color-ink-2)',
          fontWeight: 500,
          minHeight: '1.5rem',
          transition: 'opacity 0.3s ease',
        }}
      >
        {message}
      </p>
    </div>
  );
}

/* ─── Main App ─── */
function App() {
  const [screenState, setScreenState] = useState<
    'landing' | 'recording' | 'analyzing' | 'results'
  >('landing');
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loaderMessage, setLoaderMessage] = useState('Initializing pipelines...');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  /* Theme class on <html> */
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.toggle('light', theme === 'light');
  }, [theme]);

  /* API health polling */
  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/health`);
      if (res.ok) {
        const data = await res.json();
        setApiOnline(true);
        setModelLoaded(data.model_loaded);
        setIsTraining(data.is_training);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  };

  const triggerModelTraining = async () => {
    setIsTraining(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/train`, { method: 'POST' });
      if (res.ok) {
        let trained = false;
        while (!trained) {
          await new Promise(r => setTimeout(r, 2000));
          const hr = await fetch(`${BACKEND_URL}/api/v1/health`);
          if (hr.ok) {
            const hd = await hr.json();
            if (!hd.is_training) {
              trained = true;
              setModelLoaded(hd.model_loaded);
              setIsTraining(false);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setIsTraining(false);
    }
  };

  /* Cycling loader steps */
  useEffect(() => {
    if (screenState !== 'analyzing') return;
    const steps = [
      'Receiving WAV voice payload...',
      'Running downsampling & audio resampling...',
      'Executing spectral gating noise reduction...',
      'Locating speech cycles (voiced F0 segments)...',
      'Calculating cycle-by-cycle pitch Jitter...',
      'Extracting cycle-by-cycle amplitude Shimmer...',
      'Computing Harmonics-to-Noise Ratio (HNR)...',
      'Loading Wav2Vec 2.0 speech transformer...',
      'Extracting 768-dimensional neural embeddings...',
      'Applying PCA dimensionality reduction...',
      'Concatenating hybrid biomarker feature maps...',
      'Running SV Ensemble classifier model...',
      'Generating health report summary...',
    ];
    let i = 0;
    setLoaderMessage(steps[0]);
    const interval = setInterval(() => {
      if (i < steps.length - 1) { i++; setLoaderMessage(steps[i]); }
    }, 1800);
    return () => clearInterval(interval);
  }, [screenState]);

  const handleAnalysisStart = () => {
    setErrorMsg(null);
    setScreenState('analyzing');
    setLoaderMessage('Receiving WAV voice payload...');
  };

  const handleLoadHistoryRecord = (recordData: any) => {
    setAnalysisResult(recordData);
    setScreenState('results');
  };

  const handleUploadSuccess = (data: any) => {
    setAnalysisResult(data);
    setScreenState('results');
    try {
      const historyStr = localStorage.getItem('vitavoice_history') || '[]';
      const history = JSON.parse(historyStr);
      const reportUrl = data.report_url || '';
      const match = reportUrl.match(/report_([a-f0-9]+)\.pdf/);
      const fileId = match ? match[1] : Math.random().toString(36).substring(2, 10);
      const newRecord = {
        id: fileId,
        timestamp: new Date().toISOString(),
        risk_score: data.risk_score,
        risk_category: data.report.risk_category,
        certainty_label: data.report.confidence_calibration?.certainty_label || 'N/A',
        data,
      };
      localStorage.setItem(
        'vitavoice_history',
        JSON.stringify([newRecord, ...history].slice(0, 10))
      );
    } catch (err) {
      console.error('Failed to save history:', err);
    }
  };

  const handleAnalysisError = (err: string) => {
    setErrorMsg(err);
    setScreenState('recording');
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setErrorMsg(null);
    setScreenState('landing');
  };

  /* ── Derived UI helpers ── */
  const isLanding = screenState === 'landing';

  return (
    <div className="bg-scene" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── N5 Floating Pill Nav ── */}
      <div
        style={{
          position: 'fixed',
          top: '1.25rem',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 100,
          width: 'calc(100% - 2.5rem)',
          maxWidth: '900px',
        }}
      >
        <nav
          className="nav-pill"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.625rem 1.25rem',
          }}
        >
          {/* Wordmark */}
          <button
            onClick={handleReset}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.625rem',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 0,
            }}
            aria-label="Return to home"
          >
            <div
              style={{
                width: 34, height: 34,
                borderRadius: 10,
                background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-warm))',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'var(--font-display)',
                fontWeight: 900,
                fontSize: '1rem',
                color: 'oklch(10% 0.025 275)',
                boxShadow: '0 0 20px -4px var(--color-glow-cyan)',
                flexShrink: 0,
              }}
            >
              V
            </div>
            <div style={{ lineHeight: 1 }}>
              <span
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 800,
                  fontSize: '1rem',
                  color: 'var(--color-ink)',
                  letterSpacing: '-0.01em',
                }}
              >
                Vita<span style={{ color: 'var(--color-accent)' }}>Voice</span>
              </span>
              <span
                style={{
                  display: 'block',
                  fontSize: '0.5625rem',
                  color: 'var(--color-ink-3)',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.10em',
                  marginTop: 1,
                }}
              >
                Biomarker AI
              </span>
            </div>
          </button>

          {/* Right controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            {/* Theme toggle */}
            <button
              id="theme-toggle"
              onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              style={{
                width: 34, height: 34,
                borderRadius: 9,
                background: 'oklch(20% 0.028 270 / 0.60)',
                border: '1px solid var(--color-rule)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer',
                color: theme === 'dark' ? 'oklch(80% 0.18 60)' : 'var(--color-accent)',
                transition: 'all var(--dur-fast) var(--ease-out)',
              }}
              onMouseOver={e => {
                (e.currentTarget as HTMLButtonElement).style.background = 'oklch(26% 0.030 270 / 0.80)';
              }}
              onMouseOut={e => {
                (e.currentTarget as HTMLButtonElement).style.background = 'oklch(20% 0.028 270 / 0.60)';
              }}
            >
              {theme === 'dark'
                ? <Sun style={{ width: 15, height: 15 }} />
                : <Moon style={{ width: 15, height: 15 }} />
              }
            </button>

            {/* API status badge */}
            {apiOnline ? (
              <span className="badge badge-online" id="api-status-badge">
                <span
                  style={{
                    width: 6, height: 6, borderRadius: '50%',
                    background: 'var(--color-success)',
                    display: 'inline-block',
                    animation: 'recordingPulse 2s ease-in-out infinite',
                  }}
                />
                FastAPI Online
              </span>
            ) : (
              <span className="badge badge-offline" id="api-status-badge-offline">
                <ServerCrash style={{ width: 11, height: 11 }} />
                API Offline
              </span>
            )}
          </div>
        </nav>
      </div>

      {/* ── Training Banner ── */}
      {apiOnline && !modelLoaded && (
        <div
          className="training-banner animate-fade-in-up"
          style={{ marginTop: '5rem', flexShrink: 0 }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu style={{ width: 14, height: 14, color: 'var(--color-warning)', flexShrink: 0 }} />
            ML models not yet trained — trigger training on the Oxford Parkinson's dataset.
          </span>
          <button
            id="train-models-btn"
            onClick={triggerModelTraining}
            disabled={isTraining}
            className="btn-primary btn-sm"
            style={{ flexShrink: 0, whiteSpace: 'nowrap' }}
          >
            {isTraining ? (
              <>
                <RefreshCw style={{ width: 12, height: 12, animation: 'ringRotate 1s linear infinite' }} />
                <span>Training...</span>
              </>
            ) : (
              <span>Train AI Models</span>
            )}
          </button>
        </div>
      )}

      {/* ── Main Content ── */}
      <main
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          paddingTop: isLanding ? '6.5rem' : '7rem',
          paddingBottom: '3rem',
          paddingLeft: '1.25rem',
          paddingRight: '1.25rem',
        }}
      >

        {/* ═══ LANDING — Marquee Hero ═══ */}
        {screenState === 'landing' && (
          <div
            className="animate-fade-in-up"
            style={{
              width: '100%',
              maxWidth: 860,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center',
              position: 'relative',
            }}
          >
            {/* 3D Background orbs */}
            <HeroOrb
              size={480} color="var(--color-accent)"
              top="-160px" left="calc(50% - 360px)"
              blur={130} opacity={0.12}
              animClass="animate-float-a"
            />
            <HeroOrb
              size={360} color="var(--color-accent-warm)"
              top="-80px" left="calc(50% + 60px)"
              blur={110} opacity={0.10}
              animClass="animate-float-b"
            />

            {/* Grid lattice in hero */}
            <div
              aria-hidden="true"
              className="grid-lattice"
              style={{
                position: 'absolute',
                inset: '-60px -100px',
                zIndex: 0,
                maskImage: 'radial-gradient(ellipse 70% 60% at 50% 0%, black 30%, transparent 80%)',
                WebkitMaskImage: 'radial-gradient(ellipse 70% 60% at 50% 0%, black 30%, transparent 80%)',
              }}
            />

            {/* Content (above orbs) */}
            <div style={{ position: 'relative', zIndex: 1, width: '100%' }}>

              {/* Eyebrow tag */}
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.375rem 1rem',
                  background: 'oklch(20% 0.08 215 / 0.50)',
                  border: '1px solid oklch(72% 0.20 215 / 0.20)',
                  borderRadius: 9999,
                  marginBottom: '1.75rem',
                  backdropFilter: 'blur(8px)',
                }}
              >
                <Award style={{ width: 12, height: 12, color: 'var(--color-accent)' }} />
                <span
                  style={{
                    fontSize: '0.6875rem',
                    fontWeight: 700,
                    fontFamily: 'var(--font-display)',
                    color: 'var(--color-accent)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.10em',
                  }}
                >
                  Voice AI · Biomarker Screening Engine
                </span>
              </div>

              {/* Display headline */}
              <h1
                className="hero-display"
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 900,
                  fontSize: 'clamp(2.4rem, 6vw, 4.25rem)',
                  lineHeight: 1.06,
                  letterSpacing: '-0.03em',
                  color: 'var(--color-ink)',
                  marginBottom: '1.5rem',
                  fontStyle: 'normal',
                }}
              >
                Screen Health Risks
                <br />
                through{' '}
                <span className="text-gradient-cyan">Vocal Biomarker AI</span>
              </h1>

              {/* Sub-copy */}
              <p
                style={{
                  fontSize: 'clamp(0.9375rem, 2vw, 1.125rem)',
                  color: 'var(--color-ink-2)',
                  maxWidth: 600,
                  lineHeight: 1.7,
                  margin: '0 auto 2.75rem',
                }}
              >
                VitaVoice analyzes clinical acoustic features and deep Wav2Vec&nbsp;2.0
                speech embeddings to identify indicators of voice instabilities
                associated with chronic neurological disorders.
              </p>

              {/* 3D Stat tiles row */}
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '1rem',
                  justifyContent: 'center',
                  marginBottom: '2.75rem',
                  perspective: 'var(--perspective-hero)',
                }}
              >
                <StatTile label="Standard Dataset" value="Oxford Parkinson's" />
                <StatTile label="Classifier Architecture" value="SVM Ensemble" />
                <StatTile label="Neural Encoders" value="Wav2Vec 2.0" />
              </div>

              {/* Feature chips row */}
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.5rem',
                  justifyContent: 'center',
                  marginBottom: '2.75rem',
                }}
              >
                {[
                  { icon: <Mic style={{ width: 12, height: 12 }} />, label: 'Live Mic Recording' },
                  { icon: <Brain style={{ width: 12, height: 12 }} />, label: 'Deep Neural Analysis' },
                  { icon: <Zap style={{ width: 12, height: 12 }} />, label: 'Sub-second Inference' },
                  { icon: <ShieldCheck style={{ width: 12, height: 12 }} />, label: 'Clinical Biomarkers' },
                ].map(({ icon, label }) => (
                  <span
                    key={label}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.375rem',
                      padding: '0.375rem 0.875rem',
                      background: 'oklch(18% 0.028 270 / 0.60)',
                      border: '1px solid var(--color-rule)',
                      borderRadius: 9999,
                      fontSize: '0.6875rem',
                      fontWeight: 600,
                      fontFamily: 'var(--font-display)',
                      color: 'var(--color-ink-2)',
                      backdropFilter: 'blur(8px)',
                    }}
                  >
                    <span style={{ color: 'var(--color-accent)' }}>{icon}</span>
                    {label}
                  </span>
                ))}
              </div>

              {/* CTAs */}
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '1rem',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <button
                  id="start-assessment-btn"
                  onClick={() => setScreenState('recording')}
                  disabled={!apiOnline}
                  className="btn-primary glow-cyan"
                  style={{ minWidth: 220 }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', zIndex: 1 }}>
                    <Mic style={{ width: 17, height: 17 }} />
                    Start Voice Assessment
                  </span>
                </button>

                <a
                  href="#disclaimer-block"
                  id="read-disclaimer-link"
                  onClick={e => {
                    e.preventDefault();
                    document.getElementById('disclaimer-block')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                  className="btn-ghost"
                  style={{ minWidth: 180 }}
                >
                  Read Disclaimer
                </a>
              </div>

              {/* API offline warning */}
              {!apiOnline && (
                <p
                  style={{
                    marginTop: '1.25rem',
                    fontSize: '0.75rem',
                    color: 'var(--color-danger)',
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.375rem',
                  }}
                >
                  <AlertCircleIcon style={{ width: 14, height: 14 }} />
                  Please launch the FastAPI backend server first.
                </p>
              )}
            </div>
          </div>
        )}

        {/* ═══ RECORDING ═══ */}
        {screenState === 'recording' && (
          <div
            className="animate-fade-in-up"
            style={{ width: '100%', maxWidth: 680, display: 'flex', flexDirection: 'column', alignItems: 'center' }}
          >
            <button
              id="back-to-home-btn"
              onClick={handleReset}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.75rem',
                fontWeight: 600,
                color: 'var(--color-ink-3)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.375rem',
                marginBottom: '1.5rem',
                alignSelf: 'flex-start',
                transition: 'color var(--dur-fast) var(--ease-out)',
              }}
              onMouseOver={e => (e.currentTarget.style.color = 'var(--color-ink)')}
              onMouseOut={e => (e.currentTarget.style.color = 'var(--color-ink-3)')}
            >
              ← Back to home
            </button>

            {errorMsg && (
              <div
                style={{
                  width: '100%',
                  padding: '1rem 1.25rem',
                  marginBottom: '1.25rem',
                  background: 'oklch(18% 0.06 25 / 0.50)',
                  border: '1px solid oklch(40% 0.10 25 / 0.40)',
                  borderRadius: 'var(--radius-card)',
                  color: 'var(--color-danger)',
                  fontSize: '0.875rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                }}
              >
                <AlertCircleIcon style={{ width: 18, height: 18, flexShrink: 0, marginTop: 1 }} />
                <span>Error during analysis: {errorMsg}</span>
              </div>
            )}

            <AudioRecorder
              onUploadSuccess={handleUploadSuccess}
              onAnalysisStart={handleAnalysisStart}
              onAnalysisError={handleAnalysisError}
              backendUrl={BACKEND_URL}
            />
          </div>
        )}

        {/* ═══ ANALYZING ═══ */}
        {screenState === 'analyzing' && (
          <AnalyzingScreen message={loaderMessage} />
        )}

        {/* ═══ RESULTS ═══ */}
        {screenState === 'results' && analysisResult && (
          <Dashboard
            data={analysisResult}
            onReset={handleReset}
            onLoadHistoryRecord={handleLoadHistoryRecord}
            backendUrl={BACKEND_URL}
          />
        )}
      </main>

      {/* ── Ft5 Statement Footer ── */}
      <footer
        id="disclaimer-block"
        style={{
          borderTop: '1px solid var(--color-rule-subtle)',
          padding: '3rem 2rem 2.5rem',
          textAlign: 'center',
          background: 'oklch(9% 0.022 272 / 0.80)',
          backdropFilter: 'blur(12px)',
        }}
      >
        <div style={{ maxWidth: 680, margin: '0 auto' }}>
          {/* Statement line */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1.25rem',
              fontSize: '0.6875rem',
              fontWeight: 700,
              fontFamily: 'var(--font-display)',
              color: 'oklch(62% 0.22 25 / 0.80)',
              textTransform: 'uppercase',
              letterSpacing: '0.10em',
            }}
          >
            <ShieldCheck style={{ width: 14, height: 14 }} />
            Educational & Research Screening Disclaimer
          </div>

          <p
            style={{
              fontSize: '0.6875rem',
              color: 'var(--color-ink-3)',
              lineHeight: 1.8,
              marginBottom: '1.5rem',
            }}
          >
            VitaVoice is a preliminary vocal health biomarker screener designed for
            educational and wellness tracking purposes. It does not replace standard
            medical diagnostics, professional clinical opinions, or neurologists'
            assessments. Transient respiratory and throat conditions can alter voice
            features and increase risk scores. Always seek guidance from primary care
            physicians or speech pathologists regarding any medical issues.
          </p>

          <p
            style={{
              fontSize: '0.625rem',
              color: 'oklch(38% 0.016 270)',
              letterSpacing: '0.04em',
            }}
          >
            © 2026 VitaVoice Healthcare Technology Research Group. All rights reserved.
          </p>
        </div>
      </footer>

    </div>
  );
}

/* ─── Inline SVG fallback icon ─── */
function AlertCircleIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </svg>
  );
}

export default App;
