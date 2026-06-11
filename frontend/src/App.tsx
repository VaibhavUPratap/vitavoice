import { useState, useEffect, useRef } from 'react';
import { ServerCrash, Cpu, RefreshCw, Mic } from 'lucide-react';
import { AudioRecorder } from './components/AudioRecorder';
import { Dashboard } from './components/Dashboard';
import { LandingVisual } from './components/LandingVisual';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const LOADER_STEPS = [
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

function AnalyzingScreen({ message }: { message: string }) {
  return (
    <div className="analyzing reveal is-in">
      <div className="analyzing__spinner" aria-hidden="true" />
      <h3 className="analyzing__title">Analyzing Voice Biomarkers</h3>
      <p className="analyzing__meta">Running digital signal inference</p>
      <div className="analyzing__track">
        <div className="analyzing__progress" />
      </div>
      <p className="analyzing__message">{message}</p>
    </div>
  );
}

function App() {
  const [screenState, setScreenState] = useState<'landing' | 'recording' | 'analyzing' | 'results'>('landing');
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loaderMessage, setLoaderMessage] = useState('Initializing pipelines...');
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
    let i = 0;
    setLoaderMessage(LOADER_STEPS[0]);
    const interval = setInterval(() => {
      if (i < LOADER_STEPS.length - 1) {
        i += 1;
        setLoaderMessage(LOADER_STEPS[i]);
      }
    }, 1800);
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
    setLoaderMessage(LOADER_STEPS[0]);
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
              href="#disclaimer-block"
              className="nav-bar__link"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById('disclaimer-block')?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              Disclaimer
            </a>
            <a
              href="https://github.com"
              className="nav-bar__link"
              onClick={(e) => e.preventDefault()}
            >
              Documentation
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
                  Screen health risks through sustained vowel analysis.
                </h1>
                <p className="landing__lede">
                  VitaVoice combines clinical perturbation metrics with Wav2Vec&nbsp;2.0
                  embeddings to surface voice instabilities associated with neurological conditions.
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
                  <a
                    href="#disclaimer-block"
                    id="read-disclaimer-link"
                    className="btn btn--outline"
                    onClick={(e) => {
                      e.preventDefault();
                      document.getElementById('disclaimer-block')?.scrollIntoView({ behavior: 'smooth' });
                    }}
                  >
                    Read Disclaimer
                  </a>
                </div>
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
                    <p className="stat-block__value">Wav2Vec 2.0</p>
                  </div>
                </div>
              </div>
              <LandingVisual />
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

        {screenState === 'analyzing' && <AnalyzingScreen message={loaderMessage} />}

        {screenState === 'results' && analysisResult && (
          <Dashboard
            data={analysisResult}
            onReset={handleReset}
            onLoadHistoryRecord={handleLoadHistoryRecord}
            backendUrl={BACKEND_URL}
          />
        )}
      </main>

      <footer id="disclaimer-block" className="foot-dense">
        <div className="foot-dense__inner">
          <p>
            <strong>Educational &amp; research screening disclaimer.</strong>{' '}
            VitaVoice is a preliminary vocal health biomarker screener for educational and wellness
            tracking. It does not replace medical diagnostics or neurologist assessments. Transient
            respiratory conditions can alter voice features. Consult a qualified physician for
            medical concerns. © 2026 VitaVoice Healthcare Technology Research Group.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
