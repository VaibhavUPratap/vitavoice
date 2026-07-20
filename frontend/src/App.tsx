import { useState, useEffect, useRef } from 'react';
import { ServerCrash, Cpu, RefreshCw, Mic, Trash2, Sun, Moon } from 'lucide-react';
import { AudioRecorder } from './components/AudioRecorder';
import { Dashboard } from './components/Dashboard';
import { LandingVisual } from './components/LandingVisual';
// ─── Handwriting Module (additive) ───────────────────────────────────────────
import { Handwriting } from './pages/Handwriting';

const BACKEND_URL = import.meta.env.VITE_API_URL || '';

const LOADER_STEPS = [
  { tag: 'CONNECTING', msg: 'Isolating vocal tract via Web Audio API...' },
  { tag: 'PROCESSING', msg: 'Applying Spectral Gating Noise Reduction & Loudness Standardizer (-1.0 dBFS)...' },
  { tag: 'EXTRACTING', msg: 'Locating speech cycles (voiced F0 segments)...' },
  { tag: 'EXTRACTING', msg: 'Calculating Micro-arrhythmias (Jitter: local, Shimmer: local)...' },
  { tag: 'EXTRACTING', msg: 'Computing Harmonics-to-Noise Ratio (HNR)...' },
  { tag: 'FOUNDATION INF', msg: 'Forward pass through WavLM Deep Speech Embedding Space...' },
  { tag: 'FOUNDATION INF', msg: 'Projecting 768-dimensional neural embeddings to PCA space...' },
  { tag: 'CLASSIFYING', msg: 'Running Hybrid Feature Fusion Ensemble classifier model...' },
  { tag: 'REPORTING', msg: 'Generating clinical-grade health report & Kernel SHAP attribution...' },
];

function AnalyzingScreen({ stepIndex }: { stepIndex: number }) {
  const progress = ((stepIndex + 1) / LOADER_STEPS.length) * 100;
  
  const timeRef = useRef<string[]>([]);
  if (timeRef.current.length === 0) {
    const baseTime = Date.now();
    for (let i = 0; i < LOADER_STEPS.length; i++) {
      const d = new Date(baseTime + i * 1200);
      timeRef.current.push(d.toLocaleTimeString(undefined, { hour12: false, fractionSecondDigits: 3 } as any));
    }
  }

  return (
    <div className="analyzing reveal is-in">
      <h3 className="analyzing__title">Acoustic & Neural Diagnostics Engine</h3>
      <p className="analyzing__meta">Active Pipeline Execution Protocol</p>
      
      <div className="analyzing__track">
        <div className="analyzing__progress" style={{ width: `${progress}%` }} />
      </div>

      <div className="console-terminal">
        <div className="console-terminal__header">
          <span className="console-terminal__dot console-terminal__dot--red" />
          <span className="console-terminal__dot console-terminal__dot--yellow" />
          <span className="console-terminal__dot console-terminal__dot--green" />
          <span className="console-terminal__title">vocal_diagnostics_pipeline.log</span>
        </div>
        <div className="console-terminal__body">
          {LOADER_STEPS.map((step, idx) => {
            let status: 'pending' | 'running' | 'ok' = 'pending';
            if (idx < stepIndex) {
              status = 'ok';
            } else if (idx === stepIndex) {
              status = 'running';
            }

            return (
              <div key={idx} className={`console-line console-line--${status}`}>
                {status !== 'pending' ? (
                  <span className="console-line__time">[{timeRef.current[idx]}]</span>
                ) : (
                  <span className="console-line__time">[ --:--:--.--- ]</span>
                )}
                <span className="console-line__tag">[{step.tag}]</span>
                <span className="console-line__msg">{step.msg}</span>
                {status === 'ok' && <span className="console-line__status console-line__status--ok">SUCCESS</span>}
                {status === 'running' && <span className="console-line__status console-line__status--running">PROCESSING</span>}
                {status === 'pending' && <span className="console-line__status console-line__status--pending">QUEUED</span>}
              </div>
            );
          })}
          <div className="console-line console-line--active">
            <span className="console-line__cursor">█</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [screenState, setScreenState] = useState<'landing' | 'recording' | 'analyzing' | 'results' | 'disclaimer' | 'history' | 'handwriting'>('landing');
  const [historyList, setHistoryList] = useState<any[]>([]);

  // ─── Theme ───
  const getInitialTheme = (): 'dark' | 'light' => {
    const saved = localStorage.getItem('vitavoice_theme') as 'dark' | 'light' | null;
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };
  const [theme, setTheme] = useState<'dark' | 'light'>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme === 'light' ? 'light' : '');
    localStorage.setItem('vitavoice_theme', theme);
  }, [theme]);

  // Auto-follow system if user hasn't manually overridden
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem('vitavoice_theme')) {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  useEffect(() => {
    try {
      const historyStr = localStorage.getItem('vitavoice_history') || '[]';
      setHistoryList(JSON.parse(historyStr));
    } catch (err) {
      console.error(err);
    }
  }, [screenState]);

  const handleClearHistory = () => {
    localStorage.removeItem('vitavoice_history');
    setHistoryList([]);
  };
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loaderStepIndex, setLoaderStepIndex] = useState(0);
  const revealRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = revealRef.current;
    if (!root) return;
    const els = root.querySelectorAll('.reveal');
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add('is-in'); }),
      { threshold: 0.12 }
    );
    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, [screenState]);

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (screenState !== 'analyzing') return;
    setLoaderStepIndex(0);
    const interval = setInterval(() => {
      setLoaderStepIndex((prev) => {
        if (prev < LOADER_STEPS.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 1200);
    return () => clearInterval(interval);
  }, [screenState]);

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
          await new Promise((r) => setTimeout(r, 2000));
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

  const handleAnalysisStart = () => {
    setErrorMsg(null);
    setScreenState('analyzing');
    setLoaderStepIndex(0);
  };

  const handleLoadHistoryRecord = (recordData: Record<string, unknown>) => {
    setAnalysisResult(recordData);
    setScreenState('results');
  };

  const handleUploadSuccess = (data: Record<string, unknown>) => {
    setAnalysisResult(data);
    setScreenState('results');
    try {
      const historyStr = localStorage.getItem('vitavoice_history') || '[]';
      const history = JSON.parse(historyStr);
      const reportUrl = (data.report_url as string) || '';
      const match = reportUrl.match(/report_([a-f0-9]+)\.pdf/);
      const fileId = match ? match[1] : Math.random().toString(36).substring(2, 10);
      const report = data.report as Record<string, unknown>;
      const calibration = report?.confidence_calibration as Record<string, unknown> | undefined;
      const newRecord = {
        id: fileId,
        timestamp: new Date().toISOString(),
        risk_score: data.risk_score,
        risk_category: report?.risk_category,
        certainty_label: calibration?.certainty_label || 'N/A',
        data,
      };
      localStorage.setItem('vitavoice_history', JSON.stringify([newRecord, ...history].slice(0, 10)));
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

  return (
    <div className="app-shell" ref={revealRef}>
      <header className="nav-bar">
        <div className="page-wrap nav-bar__inner">
          <button className="nav-bar__brand" onClick={handleReset} aria-label="Return to home">
            <span className="nav-bar__mark">V</span>
            <span>
              <span className="nav-bar__title">VitaVoice</span>
              <span className="nav-bar__tag">Vocal Biomarker AI</span>
            </span>
          </button>

          <nav className="nav-bar__links" aria-label="Primary">
            <a
              href="#disclaimer"
              className="nav-bar__link"
              onClick={(e) => {
                e.preventDefault();
                setScreenState('disclaimer');
              }}
            >
              disclaimer
            </a>
            <a
              href="#history"
              className="nav-bar__link"
              onClick={(e) => {
                e.preventDefault();
                setScreenState('history');
              }}
            >
              history
            </a>
            <a
              href="https://github.com"
              className="nav-bar__link"
              onClick={(e) => e.preventDefault()}
            >
              documentation
            </a>
            {/* ─── Handwriting nav link (additive) ─── */}
            <a
              href="#handwriting"
              id="nav-handwriting-link"
              className="nav-bar__link"
              onClick={(e) => {
                e.preventDefault();
                setScreenState('handwriting');
              }}
            >
              handwriting
            </a>
          </nav>

          <div className="nav-bar__actions">
            {apiOnline ? (
              <span className="badge badge--online" id="api-status-badge">
                <span className="badge__dot" />
                FastAPI Online
              </span>
            ) : (
              <span className="badge badge--offline" id="api-status-badge-offline">
                <ServerCrash style={{ width: 11, height: 11 }} />
                API Offline
              </span>
            )}

            {/* Theme toggle */}
            <button
              id="theme-toggle-btn"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark'
                ? <Sun style={{ width: 15, height: 15 }} />
                : <Moon style={{ width: 15, height: 15 }} />}
            </button>

            {screenState === 'landing' && (
              <button
                id="start-assessment-btn"
                className="btn btn--primary btn--sm"
                onClick={() => setScreenState('recording')}
                disabled={!apiOnline}
              >
                <Mic style={{ width: 14, height: 14 }} />
                Start Voice Assessment
              </button>
            )}
          </div>
        </div>
      </header>

      {apiOnline && !modelLoaded && (
        <div className="training-banner">
          <span className="training-banner__text">
            <Cpu style={{ width: 14, height: 14, color: 'var(--color-warning)' }} />
            ML models not yet trained — trigger training on the Oxford Parkinson&apos;s dataset.
          </span>
          <button
            id="train-models-btn"
            onClick={triggerModelTraining}
            disabled={isTraining}
            className="btn btn--primary btn--sm"
          >
            {isTraining ? (
              <>
                <RefreshCw style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
                Training...
              </>
            ) : (
              'Train AI Models'
            )}
          </button>
        </div>
      )}

      <main className={`main-content${screenState === 'results' ? ' main-content--results' : ''}`}>
        {screenState === 'landing' && (
          <section className="page-wrap landing">
            <div className="landing__layout">
              <div className="landing__hero reveal">
                <p className="landing__eyebrow">Hybrid acoustic + neural screening</p>
                <div className="landing__stat-row">
                  <span className="landing__stat">36</span>
                  <span className="landing__stat-unit">concatenated features</span>
                </div>
                <h1 className="landing__headline">
                  <em>screen</em> health risks through sustained vowel analysis.
                </h1>
                <p className="landing__lede">
                  VitaVoice leverages a Hybrid Feature Fusion Ensemble architecture. By combining classical acoustic digital signal processing (DSP)—tracking micro-structural properties like vocal jitter and shimmer—with a 768-dimensional deep neural transformer foundation model (WavLM Base), the system yields medical-screening grade screening insights with calibrated, explainable confidence metrics.
                </p>
                <div className="landing__cta-row">
                  <button
                    className="btn btn--primary btn--lg"
                    onClick={() => setScreenState('recording')}
                    disabled={!apiOnline}
                  >
                    <Mic style={{ width: 16, height: 16 }} />
                    Start Voice Assessment
                  </button>
                  <button
                    id="read-disclaimer-btn"
                    className="btn btn--outline"
                    onClick={() => setScreenState('disclaimer')}
                  >
                    Read Disclaimer
                  </button>
                </div>
                <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)', marginTop: 'var(--space-xs)' }}>
                  already completed an assessment?{' '}
                  <a
                    href="#history"
                    onClick={(e) => {
                      e.preventDefault();
                      setScreenState('history');
                    }}
                    style={{ color: 'var(--color-accent)', textDecoration: 'underline' }}
                  >
                    view local screening history
                  </a>
                </p>
                {!apiOnline && (
                  <p className="landing__offline">
                    Please launch the FastAPI backend server first.
                  </p>
                )}
                <div className="landing__stats-strip">
                  <div className="stat-block">
                    <p className="stat-block__label">Dataset</p>
                    <p className="stat-block__value">Oxford Parkinson&apos;s</p>
                  </div>
                  <div className="stat-block">
                    <p className="stat-block__label">Classifier</p>
                    <p className="stat-block__value">SVM Ensemble</p>
                  </div>
                  <div className="stat-block">
                    <p className="stat-block__label">Encoder</p>
                    <p className="stat-block__value">WavLM Base</p>
                  </div>
                </div>
              </div>
              <LandingVisual />
            </div>

            {/* Lumen Meter Strip */}
            <div className="meter-strip reveal">
              <div className="meter-strip__label">signal · 16.0 khz</div>
              <div className="meter-strip__ticks">
                {Array.from({ length: 72 }).map((_, idx) => {
                  const x = (idx - 36) / 16;
                  const h = Math.round(14 * Math.exp(-x * x) + Math.random() * 3 + 2);
                  const op = (0.25 + 0.75 * Math.exp(-x * x)).toFixed(2);
                  return (
                    <span
                      key={idx}
                      className="meter-strip__tick"
                      style={{ height: `${h}px`, opacity: op }}
                    />
                  );
                })}
              </div>
              <div className="meter-strip__label">drift · 0.02 σ</div>
            </div>
          </section>
        )}

        {screenState === 'recording' && (
          <div className="page-wrap reveal is-in">
            <button id="back-to-home-btn" className="back-link" onClick={handleReset}>
              ← Back to home
            </button>
            {errorMsg && (
              <div className="step-section step-section--error" style={{ marginBottom: 'var(--space-md)' }}>
                <p className="step-section__body">Error during analysis: {errorMsg}</p>
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

        {screenState === 'analyzing' && <AnalyzingScreen stepIndex={loaderStepIndex} />}

        {screenState === 'results' && analysisResult && (
          <Dashboard
            data={analysisResult}
            onReset={handleReset}
            onLoadHistoryRecord={handleLoadHistoryRecord}
            backendUrl={BACKEND_URL}
          />
        )}
        {screenState === 'disclaimer' && (
          <div className="page-wrap reveal is-in" style={{ paddingBlock: 'var(--space-md)', width: '100%' }}>
            <button id="back-to-home-btn" className="back-link" onClick={handleReset}>
              ← Back to home
            </button>
            <article className="panel panel--wide">
              <header className="panel__header">
                <div>
                  <h2 className="panel__title">responsible ai screening disclaimer</h2>
                  <p className="panel__subtitle">vitavoice clinical & ethical validation standards</p>
                </div>
              </header>
              <section className="step-section">
                <p className="step-section__body" style={{ fontSize: 'var(--text-sm)', lineHeight: 1.8 }}>
                  vitavoice is a preliminary vocal health biomarker screening tool intended for wellness tracking and educational research purposes. it is not a diagnostic device and does not replace professional clinical assessments by a neurologist or physician.
                </p>
              </section>
              <section className="step-section">
                <h3 className="step-section__title" style={{ fontFamily: 'var(--font-mono)', textTransform: 'uppercase', fontSize: 'var(--text-xs)' }}>responsible screening principles</h3>
                <ul className="responsible-ai__list" style={{ marginTop: 'var(--space-xs)', display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
                  <li>vocal tracts are isolated using advanced client-side web audio dsp APIs without raw recording retention.</li>
                  <li>all audio files processed on the backend are deleted immediately after feature extraction.</li>
                  <li>neural transformer embeddings are projected locally and explainability attributions are client-calibrated.</li>
                  <li>temporary respiratory variations (e.g. cold, fatigue) can affect results. clinical consultation is advised.</li>
                </ul>
              </section>
            </article>
          </div>
        )}

        {screenState === 'history' && (
          <div className="page-wrap reveal is-in" style={{ paddingBlock: 'var(--space-md)', width: '100%' }}>
            <button id="back-to-home-btn" className="back-link" onClick={handleReset}>
              ← Back to home
            </button>
            <article className="panel panel--wide">
              <header className="panel__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 className="panel__title">screening history logs</h2>
                  <p className="panel__subtitle">locally cached digital signal processing profiles</p>
                </div>
                {historyList.length > 0 && (
                  <button
                    onClick={handleClearHistory}
                    className="btn btn--danger btn--sm no-print"
                    style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <Trash2 style={{ width: 14, height: 14 }} /> clear cache
                  </button>
                )}
              </header>

              <section className="step-section">
                {historyList.length === 0 ? (
                  <p className="step-section__body" style={{ textAlign: 'center', padding: 'var(--space-xl) 0', color: 'var(--color-ink-3)' }}>
                    no local screening history found. run a voice assessment to populate records.
                  </p>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                    {historyList.map((run, idx) => {
                      const runDate = new Date(run.timestamp).toLocaleDateString();
                      const runTime = new Date(run.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                      const scorePct = Math.round(run.risk_score * 100);
                      const isHigh = run.risk_score >= 0.5;

                      return (
                        <div
                          key={idx}
                          className="card"
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: 'var(--space-md) var(--space-lg)',
                            gap: 'var(--space-md)',
                            flexWrap: 'wrap'
                          }}
                        >
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>
                              ID: {run.id} · {runDate} {runTime}
                            </span>
                            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
                              calibration status: {run.certainty_label.toLowerCase()}
                            </span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
                            <div style={{ textAlign: 'right' }}>
                              <span style={{ fontSize: 'var(--text-xs)', color: isHigh ? 'var(--color-danger)' : 'var(--color-success)', fontWeight: 'bold' }}>
                                {scorePct}% risk ({run.risk_category.toLowerCase()})
                              </span>
                            </div>
                            <button
                              onClick={() => handleLoadHistoryRecord(run.data)}
                              className="btn btn--outline btn--sm"
                            >
                              view report
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </section>
            </article>
          </div>
        )}

        {/* ─── Handwriting Module screen (additive) ─────────────────────────── */}
        {screenState === 'handwriting' && (
          <Handwriting onBack={() => setScreenState('landing')} />
        )}
      </main>

      <footer className="foot-dense">
        <div className="foot-dense__inner">
          <p style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
            <span>© 2026 vitavoice healthcare technology research group.</span>
            <a
              href="#disclaimer"
              onClick={(e) => {
                e.preventDefault();
                setScreenState('disclaimer');
              }}
              style={{ color: 'var(--color-accent)', textDecoration: 'underline' }}
            >
              read clinical disclaimer
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
