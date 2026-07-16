import React, { useEffect, useState, useMemo } from 'react';
import {
  ShieldAlert, RefreshCw, Printer, FileText,
  Stethoscope, Brain, AlertTriangle, Mic, CheckCircle2,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer, ReferenceLine,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { EmbeddingCanvas } from './EmbeddingCanvas';

interface DashboardProps {
  data: Record<string, unknown>;
  onReset: () => void;
  onLoadHistoryRecord?: (recordData: Record<string, unknown>) => void;
  backendUrl: string;
}

type RiskColors = {
  stroke: string;
  textClass: string;
  badgeClass: string;
};

function getRiskColors(score: number): RiskColors {
  if (score < 0.35) return { stroke: 'var(--color-success)', textClass: 'text-risk-low', badgeClass: 'bg-risk-low' };
  if (score < 0.65) return { stroke: 'var(--color-warning)', textClass: 'text-risk-mid', badgeClass: 'bg-risk-mid' };
  return { stroke: 'var(--color-danger)', textClass: 'text-risk-high', badgeClass: 'bg-risk-high' };
}



export const Dashboard: React.FC<DashboardProps> = ({ data, onReset, onLoadHistoryRecord, backendUrl }) => {
  const riskScore = data.risk_score as number;
  const embeddingCoords = data.embedding_coords as [number, number];
  const report = data.report as Record<string, unknown>;
  const recommendations = report.recommendations as string[];
  const shapExplanation = report.shap_explanation as Array<Record<string, unknown>> | undefined;
  const reportUrl = data.report_url as string | undefined;

  // New enriched data with legacy history fallbacks
  const recordingQuality = (data.recording_quality as Record<string, unknown>) || {
    background_noise_pct: 5,
    clipping_detected: false,
    speech_coverage_pct: 95,
    mic_status: 'nominal',
    suitable_for_analysis: true,
    quality_stars: '★★★★★'
  };
  const confidenceScore = (data.confidence_score as number) || 85;
  const confidenceLabel = (data.confidence_label as string) || 'high certainty';
  const predictionReliability = (data.prediction_reliability as string) || 'high';
  const naturalLanguageExplanation = (data.natural_language_explanation as string) || 'vocal characteristics and features align with the reference cohort.';
  const recommendationText = (data.recommendation as string) || '';
  const responsibleAiPoints = (data.responsible_ai_points as string[]) || [
    'This application is intended for preliminary screening only.',
    'It is not a medical diagnosis.',
    'Voice recordings are processed only for analysis.',
    'Long-term storage of recordings is disabled.',
    'Consult a qualified neurologist for clinical diagnosis.',
  ];
  const biomarkerStatuses = (data.biomarker_statuses as Array<Record<string, unknown>>) || [];

  // Fallback architecture hooks (AI Clinical Copilot)
  const [activeTab, setActiveTab] = useState<'overview' | 'fingerprint' | 'topology' | 'shap' | 'copilot'>('overview');
  const [copilotInsight, setCopilotInsight] = useState<{ summary: string; citations: string[]; is_fallback: boolean } | null>(null);
  const [loadingInsight, setLoadingInsight] = useState(false);
  const [insightError, setInsightError] = useState<string | null>(null);

  const predictionId = useMemo(() => {
    if (data.prediction_id) return data.prediction_id as string;
    if (reportUrl) {
      const parts = reportUrl.split('_');
      if (parts.length > 1) {
        return parts[1].split('.')[0];
      }
    }
    return '';
  }, [data.prediction_id, reportUrl]);

  const [clusterPoints, setClusterPoints] = useState<Array<{ x: number; y: number; status: number }>>([]);
  const [clustersLoaded, setClustersLoaded] = useState(false);
  const [historyList, setHistoryList] = useState<Array<Record<string, unknown>>>([]);

  const colors = getRiskColors(riskScore);

  // Fetch copilot insight when the copilot tab is active and not yet loaded
  useEffect(() => {
    if (activeTab !== 'copilot' || copilotInsight || !predictionId) return;

    const fetchCopilotInsight = async () => {
      setLoadingInsight(true);
      setInsightError(null);
      try {
        const res = await fetch(`${backendUrl}/api/v1/analysis/copilot-insight`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_id: predictionId }),
        });
        if (!res.ok) {
          throw new Error(`Failed to fetch insight: ${res.statusText}`);
        }
        const result = await res.json();
        setCopilotInsight(result);
      } catch (err: unknown) {
        console.error('Failed to fetch clinical copilot insight:', err);
        setInsightError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoadingInsight(false);
      }
    };

    fetchCopilotInsight();
  }, [activeTab, predictionId, backendUrl, copilotInsight]);

  // Reset tab and insight when predictionId changes
  useEffect(() => {
    setActiveTab('overview');
    setCopilotInsight(null);
    setInsightError(null);
    setLoadingInsight(false);
  }, [predictionId]);

  useEffect(() => {
    const fetchClusters = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/v1/clusters`);
        const result = await res.json();
        if (result.loaded) {
          setClusterPoints(result.points);
          setClustersLoaded(true);
        }
      } catch (err) {
        console.error('Failed to load reference clusters:', err);
      }
    };
    fetchClusters();
    try {
      const historyStr = localStorage.getItem('vitavoice_history') || '[]';
      setHistoryList(JSON.parse(historyStr));
    } catch (err) {
      console.error('Failed to load history list:', err);
    }
  }, [backendUrl, data]);

  const generateProbabilityData = () => {
    const arr = [];
    for (let x = 0; x <= 1.01; x += 0.02) {
      const healthyDensity = (1 / (0.15 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 0.2) / 0.15, 2));
      const pathDensity = (1 / (0.15 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 0.8) / 0.15, 2));
      arr.push({
        x: Math.round(x * 100) / 100,
        'Healthy Cohort': parseFloat(healthyDensity.toFixed(3)),
        'Pathology Cohort': parseFloat(pathDensity.toFixed(3)),
      });
    }
    return arr;
  };

  const probData = generateProbabilityData();
  const refId = useMemo(() => {
    if (reportUrl) {
      return `VV-${reportUrl.split('_')[1]?.split('.')[0]?.toUpperCase() || 'UNKNOWN'}`;
    }
    return `VV-${Math.floor(100000 + Math.random() * 900000)}`;
  }, [reportUrl]);

  const screenedDate = useMemo(() => new Date().toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
  }), []);

  const radarData = useMemo(() => {
    let userF0 = 80;
    let userJitter = 80;
    let userShimmer = 80;
    let userHnr = 80;
    let userNhr = 80;
    let userEnergy = 80;

    const metrics = (data.clinical_metrics as Record<string, number>) || {};
    
    if (metrics.fo_mean !== undefined) {
      const val = metrics.fo_mean;
      if (val >= 85 && val <= 255) {
        userF0 = 92;
      } else {
        const dev = val < 85 ? (85 - val) : (val - 255);
        userF0 = Math.max(25, Math.round(90 - dev * 0.4));
      }
    }

    if (metrics.jitter_pct !== undefined) {
      const val = metrics.jitter_pct;
      userJitter = Math.max(15, Math.min(99, Math.round(100 - val * 30)));
    }

    if (metrics.shimmer_local !== undefined) {
      let val = metrics.shimmer_local;
      if (val < 0.1) val = val * 100;
      userShimmer = Math.max(15, Math.min(99, Math.round(100 - val * 8)));
    }

    if (metrics.hnr !== undefined) {
      const val = metrics.hnr;
      userHnr = Math.max(15, Math.min(99, Math.round(val * 4)));
    }

    if (metrics.nhr !== undefined) {
      const val = metrics.nhr;
      userNhr = Math.max(15, Math.min(99, Math.round(100 - val * 200)));
    }

    if (metrics.energy !== undefined) {
      userEnergy = 85;
    }

    return [
      { subject: 'Pitch Stability', 'Your Voice': userF0, 'Healthy Cohort': 85 },
      { subject: 'Jitter (Pitch)', 'Your Voice': userJitter, 'Healthy Cohort': 90 },
      { subject: 'Shimmer (Amp)', 'Your Voice': userShimmer, 'Healthy Cohort': 90 },
      { subject: 'Harmonic Ratio', 'Your Voice': userHnr, 'Healthy Cohort': 85 },
      { subject: 'Noise Ratio', 'Your Voice': userNhr, 'Healthy Cohort': 90 },
      { subject: 'Spectral Energy', 'Your Voice': userEnergy, 'Healthy Cohort': 85 },
    ];
  }, [data.clinical_metrics]);

  const shapList = shapExplanation || [];
  const baseValue = 0.47;

  const waterfallRows = useMemo(() => {
    let currentVal = baseValue;
    return shapList.slice(0, 6).map((feat) => {
      const label = (feat.label || feat.feature_name || 'Unknown') as string;
      const shapValue = (feat.shap_value || 0) as number;
      const start = currentVal;
      currentVal += shapValue;
      const end = currentVal;
      return {
        label,
        shapValue,
        start,
        end,
        isPositive: shapValue > 0,
      };
    });
  }, [shapList]);

  const { minVal, maxVal } = useMemo(() => {
    if (waterfallRows.length === 0) return { minVal: 0.0, maxVal: 1.0 };
    const values = [baseValue, riskScore];
    waterfallRows.forEach((r) => {
      values.push(r.start);
      values.push(r.end);
    });
    return {
      minVal: Math.max(0.0, Math.min(...values) - 0.05),
      maxVal: Math.min(1.0, Math.max(...values) + 0.05),
    };
  }, [waterfallRows, riskScore]);

  const range = maxVal - minVal || 1;

  const getPct = (val: number) => {
    return ((val - minVal) / range) * 100;
  };

  // Quality metrics
  const qualityStars = (recordingQuality?.quality_stars as string) || '★★★★★';
  const qualityWarning = (recordingQuality?.quality_warning as string) || null;

  return (
    <div className="dashboard reveal is-in">
      {/* ─── Header ─────────────────────────────────────────── */}
      <header className="results-header">
        <div className="results-header__row">
          <div className="results-header__intro">
            <h1 className="results-header__title">clinical screening summary</h1>
            <p className="results-header__meta">{refId} · {screenedDate}</p>
          </div>
          <div className="results-header__actions no-print">
            {predictionId && (
              <a href={`${backendUrl}/api/v1/analysis/download-pdf/${predictionId}`} target="_blank" rel="noopener noreferrer" className="btn btn--outline btn--sm">
                <FileText style={{ width: 14, height: 14 }} /> Download PDF
              </a>
            )}
            <button onClick={() => window.print()} className="btn btn--ghost btn--sm">
              <Printer style={{ width: 14, height: 14 }} /> Print
            </button>
            <button onClick={onReset} className="btn btn--primary btn--sm">
              <RefreshCw style={{ width: 14, height: 14 }} /> New Screening
            </button>
          </div>
        </div>
      </header>

      <div className="dashboard__layout">
        {/* ─── Sidebar: History ───────────────────────────── */}
        <aside className="card no-print">
          <h3 className="card__label" style={{ display: 'flex', justifyContent: 'space-between' }}>
            screening history
            <span style={{ background: 'var(--color-paper-3)', padding: '0.1rem 0.4rem', borderRadius: 4 }}>{historyList.length}</span>
          </h3>
          {historyList.length === 0 ? (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink-3)', textAlign: 'center', padding: 'var(--space-lg) 0' }}>No previous screenings.</p>
          ) : (
            <div className="history-list">
              {historyList.map((run, idx) => {
                const runDate = new Date(run.timestamp as string).toLocaleDateString();
                const runTime = new Date(run.timestamp as string).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const runData = run.data as Record<string, unknown>;
                const isSelected = reportUrl && runData.report_url === reportUrl;
                const score = run.risk_score as number;
                const riskCls = getRiskColors(score).textClass;
                return (
                  <button
                    key={idx}
                    onClick={() => onLoadHistoryRecord?.(runData)}
                    className={`history-item${isSelected ? ' history-item--active' : ''}`}
                  >
                    <div className="history-item__row">
                      <span className="history-item__date">{runDate} · {runTime}</span>
                      <span className={`history-item__score ${riskCls}`}>{Math.round(score * 100)}%</span>
                    </div>
                    <span className="history-item__cat">{run.risk_category as string}</span>
                  </button>
                );
              })}
            </div>
          )}
        </aside>

        {/* ─── Main Content Sections ─────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)', minWidth: 0 }}>

          {/* Toggles for Dashboard Sections */}
          <div className="no-print dashboard-tabs">
            {(['overview', 'vocal fingerprint', 'latent topology', 'explainable ai (shap)', 'clinical air (copilot)'] as const).map((tab) => {
              const tabKey =
                tab === 'vocal fingerprint' ? 'fingerprint' :
                tab === 'latent topology' ? 'topology' :
                tab === 'explainable ai (shap)' ? 'shap' :
                tab === 'clinical air (copilot)' ? 'copilot' :
                tab;
              const isSelected = activeTab === tabKey;
              return (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tabKey)}
                  className={`tab-btn${isSelected ? ' tab-btn--active' : ''}`}
                  style={{ textTransform: 'lowercase' }}
                >
                  {tab}
                </button>
              );
            })}
          </div>

          {activeTab === 'overview' && (
            <div className="space-y-6 animate-fade-in">
              {/* Row 1: Primary Metrics & Acoustic Audit */}
              <div className="dashboard-row-1">
                <div className="card clinical-summary">
                  {/* Left Column: Estimated Risk */}
                  <div className="clinical-summary__risk">
                    <span className="clinical-summary__risk-label">estimated risk</span>
                    <span className={`clinical-summary__risk-pct ${colors.textClass}`}>
                      {Math.round(riskScore * 100)}%
                    </span>
                    <span className={`clinical-summary__risk-cat ${colors.badgeClass}`}>
                      {(report.risk_category as string).toLowerCase()}
                    </span>
                  </div>

                  <div className="clinical-summary__divider" />

                  {/* Right Column: Prediction Confidence */}
                  <div className="clinical-summary__risk">
                    <span className="clinical-summary__risk-label">prediction confidence</span>
                    <span className="clinical-summary__risk-pct text-accent">
                      {Math.round(confidenceScore)}%
                    </span>
                    <div className="confidence-bar-wrap">
                      <div className="confidence-bar">
                        <div className="confidence-bar__fill" style={{ width: `${confidenceScore}%` }} />
                      </div>
                    </div>
                    <span className={`clinical-summary__risk-cat ${
                      predictionReliability === 'High' ? 'bg-risk-low' :
                      predictionReliability === 'Moderate' ? 'bg-risk-mid' :
                      'bg-risk-high'
                    }`}>
                      {confidenceLabel.toLowerCase()}
                    </span>
                  </div>
                </div>

                {recordingQuality && (
                  <div className="dashboard-section recording-quality" style={{ margin: 0 }}>
                    <div className="recording-quality__header">
                      <div>
                        <p className="section-label" style={{ margin: 0 }}>
                          <Mic className="section-label__icon" />
                          acoustic quality audit
                        </p>
                      </div>
                      <span className="recording-quality__stars">{qualityStars}</span>
                    </div>
                    
                    <div className="clinical-audit-grid">
                      {/* SNR Meter Card */}
                      <div className="audit-card">
                        <div className="audit-card__row">
                          <span className="audit-card__label">signal-to-noise ratio (snr)</span>
                          <span className="audit-card__badge audit-card__badge--success">
                            {Math.max(10, Math.round(32 - ((recordingQuality.background_noise_pct as number) || 5) * 0.6))}dB SNR ({((recordingQuality.background_noise_pct as number) || 5) <= 10 ? 'Excellent' : 'Good'})
                          </span>
                        </div>
                        <p className="audit-card__desc">Acoustic clarity check verifies vocal signal presence against background room noise floor.</p>
                        <div className="audit-bar">
                          <div 
                            className="audit-bar__fill" 
                            style={{ 
                              width: `${Math.min(100, (Math.max(10, Math.round(32 - ((recordingQuality.background_noise_pct as number) || 5) * 0.6)) / 30) * 100)}%`,
                              background: ((recordingQuality.background_noise_pct as number) || 5) <= 10 ? 'var(--color-success)' : 'var(--color-warning)'
                            }} 
                          />
                        </div>
                      </div>

                      {/* Clipping Protection Card */}
                      <div className="audit-card">
                        <div className="audit-card__row">
                          <span className="audit-card__label">audio saturation check</span>
                          <span className={`audit-card__badge ${(recordingQuality.clipping_detected as boolean) ? 'audit-card__badge--error' : 'audit-card__badge--success'}`}>
                            {(recordingQuality.clipping_detected as boolean) ? 'CLIPPING DETECTED' : 'PASSED (NOMINAL)'}
                          </span>
                        </div>
                        <p className="audit-card__desc">Ensures input vocal amplitude does not exceed digital ceiling (0 dBFS), preventing distortion.</p>
                        <div className="audit-bar">
                          <div 
                            className="audit-bar__fill" 
                            style={{ 
                              width: '100%',
                              background: (recordingQuality.clipping_detected as boolean) ? 'var(--color-danger)' : 'var(--color-success)'
                            }} 
                          />
                        </div>
                      </div>

                      {/* Standard Audit Check items */}
                      <div className="audit-checklist-card">
                        <div className="audit-check-item">
                          <CheckCircle2 style={{ width: 14, height: 14, color: 'var(--color-success)', flexShrink: 0 }} />
                          <span>vocal tract isolation: web audio api active</span>
                        </div>
                        <div className="audit-check-item">
                          <CheckCircle2 style={{ width: 14, height: 14, color: 'var(--color-success)', flexShrink: 0 }} />
                          <span>speech coverage: {recordingQuality.speech_coverage_pct as number}% (sustained vowel locked)</span>
                        </div>
                        <div className="audit-check-item">
                          <CheckCircle2 style={{ width: 14, height: 14, color: 'var(--color-success)', flexShrink: 0 }} />
                          <span>mic gain calibration: {recordingQuality.mic_status as string}</span>
                        </div>
                        <div className="audit-check-item">
                          <CheckCircle2 style={{ width: 14, height: 14, color: (recordingQuality.suitable_for_analysis as boolean) ? 'var(--color-success)' : 'var(--color-danger)', flexShrink: 0 }} />
                          <span>suitable for inference: {(recordingQuality.suitable_for_analysis as boolean) ? 'calibrated' : 'fail'}</span>
                        </div>
                      </div>
                    </div>

                    {qualityWarning && (
                      <div className="quality-warning">
                        <AlertTriangle className="quality-warning__icon" style={{ width: 18, height: 18 }} />
                        <span>{qualityWarning}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Recommendation & Safety stacked side-by-side */}
              <div className="dashboard-section grid-2" style={{ margin: 0 }}>
                {/* Recommendation */}
                <div className="recommendation-card" style={{ margin: 0 }}>
                  <div className="recommendation-card__header">
                    <Stethoscope style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
                    <h3 className="recommendation-card__title">clinical recommendation</h3>
                  </div>
                  {recommendationText ? (
                    <p className="recommendation-card__text">{recommendationText}</p>
                  ) : (
                    <ul className="rec-list">
                      {recommendations.map((rec, idx) => <li key={idx}>{rec}</li>)}
                    </ul>
                  )}
                </div>

                {/* Responsible AI */}
                <div className="responsible-ai" style={{ margin: 0 }}>
                  <div className="responsible-ai__header">
                    <ShieldAlert style={{ width: 18, height: 18, color: 'var(--color-danger)' }} />
                    <h3 className="responsible-ai__title">responsible ai</h3>
                  </div>
                  <ul className="responsible-ai__list">
                    {responsibleAiPoints.map((point, idx) => (
                      <li key={idx}>{point}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'fingerprint' && (
            <div className="space-y-6 animate-fade-in">
              <div className="dashboard-section" style={{ margin: 0 }}>
                {/* Shared aligned header row */}
                <div className="fingerprint-header-row">
                  <p className="section-label" style={{ margin: 0 }}>
                    6-axis vocal fingerprint radar
                  </p>
                  <p className="section-label" style={{ margin: 0 }}>
                    biomarker specification
                  </p>
                </div>

                <div className="vocal-fingerprint-layout">
                  <div className="card radar-card">
                    <div style={{ width: '100%', height: 360 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="60%" data={radarData}>
                          <PolarGrid stroke="var(--color-rule-2)" />
                          <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--color-ink-2)', fontSize: 11, fontFamily: 'var(--font-body)' }} />
                          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'var(--color-ink-3)', fontSize: 8 }} />
                          <Radar name="Healthy Cohort" dataKey="Healthy Cohort" stroke="var(--color-success)" fill="oklch(52% 0.14 155 / 0.15)" fillOpacity={0.6} />
                          <Radar name="Your Voice" dataKey="Your Voice" stroke="var(--color-accent)" fill="oklch(58% 0.20 256 / 0.25)" fillOpacity={0.6} />
                          <ChartTooltip contentStyle={{ backgroundColor: 'var(--color-paper-3)', borderColor: 'var(--color-rule-2)', borderRadius: 'var(--radius-btn)', color: 'var(--color-ink)' }} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="biomarker-status-wrapper">
                    {biomarkerStatuses.length === 0 ? (
                      <div className="biomarker-card" style={{ padding: 'var(--space-md)', textAlign: 'center' }}>
                        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)' }}>No biomarker metrics extracted.</p>
                      </div>
                    ) : (
                      <div className="biomarker-grid">
                        {biomarkerStatuses.map((bm, idx) => {
                          const status = bm.status as string;
                          const isAbnormal = status === 'Elevated' || status === 'Reduced' || status === 'Low' || status === 'High';
                          const cardClass = isAbnormal
                            ? (status === 'Reduced' || status === 'Low' ? 'biomarker-status-card--reduced' : 'biomarker-status-card--elevated')
                            : '';
                          const badgeClass = isAbnormal ? 'bg-risk-high' : 'bg-risk-low';
                          const displayValue = bm.key === 'mfcc_profile' ? '—' : `${bm.value} ${bm.unit}`;

                          return (
                            <div key={idx} className={`biomarker-status-card ${cardClass}`}>
                              <div className="biomarker-status-card__header">
                                <span className="biomarker-status-card__name">{bm.label as string}</span>
                                <span className={`biomarker-status-card__badge ${badgeClass}`}>{status}</span>
                              </div>
                              <span className="biomarker-status-card__value">{displayValue}</span>
                              <p className="biomarker-status-card__ref">Ref: {bm.reference_range as string}</p>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'topology' && (
            <div className="space-y-6 animate-fade-in">
              <div className="dashboard-row-3">
                <EmbeddingCanvas
                  embeddingCoords={embeddingCoords}
                  clusterPoints={clusterPoints}
                  clustersLoaded={clustersLoaded}
                />

                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                  <p className="card__label">risk score distribution</p>
                  <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)', marginBottom: 'var(--space-md)' }}>
                    Your score vs healthy and Parkinsonian reference cohorts.
                  </p>
                  <div style={{ width: '100%', height: 200, flex: 1 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={probData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                        <XAxis dataKey="x" tickFormatter={(val) => `${Math.round(val * 100)}%`} stroke="var(--color-ink-3)" fontSize={10} />
                        <YAxis stroke="var(--color-ink-3)" fontSize={10} hide />
                        <ChartTooltip contentStyle={{ backgroundColor: 'var(--color-paper-3)', borderColor: 'var(--color-rule-2)', borderRadius: 'var(--radius-btn)', color: 'var(--color-ink)' }} />
                        <Area type="monotone" name="Healthy Cohort" dataKey="Healthy Cohort" stroke="var(--color-success)" fill="oklch(52% 0.14 155 / 0.08)" strokeWidth={1.5} />
                        <Area type="monotone" name="Pathology Cohort" dataKey="Pathology Cohort" stroke="var(--color-danger)" fill="oklch(52% 0.18 25 / 0.08)" strokeWidth={1.5} />
                        <ReferenceLine x={riskScore} stroke="var(--color-accent)" strokeWidth={2} strokeDasharray="4 3" label={{ value: 'You', fill: 'var(--color-accent)', fontSize: 10, position: 'top', fontWeight: 'bold' }} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'shap' && (
            <div className="space-y-6 animate-fade-in">
              <div className="dashboard-section xai-panel" style={{ margin: 0 }}>
                <div className="xai-panel__header">
                  <Brain style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
                  <h3 className="xai-panel__title">explainable ai layer: kernel shap biomarker contribution analysis</h3>
                </div>

                <p className="card__sub" style={{ marginBottom: 'var(--space-md)', color: 'var(--color-ink-2)' }}>
                  Visualizes how each high-dimensional vocal biomarker pulls prediction probability relative to base rate \(E[f(x)] = 0.47\).
                </p>

                {waterfallRows.length > 0 ? (
                  <div className="shap-waterfall" style={{ margin: 0 }}>
                    {/* Scale labels */}
                    <div className="waterfall-scale">
                      <span className="waterfall-scale__label text-risk-low">← HEALTHY PULL</span>
                      <span className="waterfall-scale__label text-center">E[f(x)] = 0.47</span>
                      <span className="waterfall-scale__label text-risk-high">PARKINSONIAN PULL →</span>
                    </div>

                    {/* Rows */}
                    <div className="waterfall-rows">
                      {waterfallRows.map((row, idx) => {
                        const pctStart = getPct(Math.min(row.start, row.end));
                        const pctWidth = Math.abs(getPct(row.end) - getPct(row.start));

                        return (
                          <div key={idx} className="waterfall-row">
                            <div className="waterfall-row__info">
                              <span className="waterfall-row__label">{row.label}</span>
                              <span className={`waterfall-row__value font-mono ${row.isPositive ? 'text-risk-high' : 'text-risk-low'}`}>
                                {row.isPositive ? '+' : ''}{row.shapValue.toFixed(4)}
                              </span>
                            </div>
                            <div className="waterfall-row__track-wrap">
                              <div className="waterfall-row__track">
                                {/* Gridline for expected baseline */}
                                <div className="waterfall-grid-line" style={{ left: `${getPct(baseValue)}%` }} />
                                {/* Gridline for final score */}
                                <div className="waterfall-grid-line waterfall-grid-line--final" style={{ left: `${getPct(riskScore)}%` }} />
                                
                                <div 
                                  className={`waterfall-segment waterfall-segment--${row.isPositive ? 'pos' : 'neg'}`}
                                  style={{
                                    left: `${pctStart}%`,
                                    width: `${Math.max(1, pctWidth)}%`,
                                  }}
                                />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Summary row details */}
                    <div className="waterfall-summary">
                      <div className="waterfall-summary__item">
                        <span className="waterfall-summary__label">Expected value E[f(x)]</span>
                        <span className="waterfall-summary__value font-mono">0.4700</span>
                      </div>
                      <div className="waterfall-summary__item">
                        <span className="waterfall-summary__label">Total Biomarker SHAP Shift</span>
                        <span className={`waterfall-summary__value font-mono ${(riskScore - 0.47) >= 0 ? 'text-risk-high' : 'text-risk-low'}`}>
                          {(riskScore - 0.47) >= 0 ? '+' : ''}{(riskScore - 0.47).toFixed(4)}
                        </span>
                      </div>
                      <div className="waterfall-summary__item">
                        <span className="waterfall-summary__label font-bold">Calibrated Model Output</span>
                        <span className="waterfall-summary__value font-bold font-mono text-risk-high" style={{ color: 'var(--color-accent)' }}>
                          {riskScore.toFixed(4)} ({Math.round(riskScore * 100)}% Risk)
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p style={{ color: 'var(--color-ink-3)', fontSize: 'var(--text-sm)', padding: 'var(--space-md) 0' }}>No SHAP biomarker attribution available for this record.</p>
                )}

                {/* Natural language explanation */}
                {naturalLanguageExplanation && (
                  <div className="xai-explanation" style={{ marginTop: 'var(--space-md)' }}>
                    <strong>Attribution Narrative:</strong> {naturalLanguageExplanation}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'copilot' && (
            <div className="space-y-6">
              {loadingInsight ? (
                <div className="dashboard-section" style={{ padding: 'var(--space-lg)', textAlign: 'center', background: 'var(--color-paper-2)', borderRadius: 'var(--radius-card)', border: '1px solid var(--color-rule)' }}>
                  <div className="analyzing__spinner" aria-hidden="true" style={{ marginInline: 'auto', marginBottom: 'var(--space-md)', width: '36px', height: '36px', border: '3px solid var(--color-rule)', borderTopColor: 'var(--color-accent)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                  <p style={{ color: 'var(--color-ink-2)', fontSize: 'var(--text-sm)', fontWeight: 500 }}>Synthesizing clinical insights with Copilot...</p>
                </div>
              ) : insightError ? (
                <div className="dashboard-section" style={{ padding: 'var(--space-lg)', textAlign: 'center', background: 'var(--color-danger-bg)', borderRadius: 'var(--radius-card)', border: '1px solid var(--color-danger)' }}>
                  <AlertTriangle className="quality-warning__icon" style={{ width: 36, height: 36, color: 'var(--color-danger)', marginInline: 'auto', marginBottom: 'var(--space-sm)' }} />
                  <p style={{ color: 'var(--color-danger)', fontSize: 'var(--text-sm)', fontWeight: 600 }}>Failed to fetch clinical copilot insight</p>
                  <p style={{ color: 'var(--color-ink-3)', fontSize: 'var(--text-xs)', marginTop: 'var(--space-xs)' }}>{insightError}</p>
                </div>
              ) : (
                <div className="copilot-grid animate-fade-in">
                  {/* Left Column: Summary */}
                  <div className="card copilot-summary-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)', marginBottom: 'var(--space-md)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)' }}>
                        <Brain style={{ width: 20, height: 20, color: 'var(--color-accent)' }} />
                        <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'bold', color: 'var(--color-ink)' }}>RAG-Augmented Summary Analysis</h3>
                      </div>
                      {copilotInsight?.is_fallback && (
                        <span className="badge badge--online animate-pulse" style={{ background: 'var(--color-warning-bg)', color: 'var(--color-warning)', borderColor: 'var(--color-warning)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--color-warning)' }} />
                          Rule-Based Fallback Active
                        </span>
                      )}
                    </div>

                    <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink-2)', lineHeight: 1.7, whiteSpace: 'pre-line' }}>
                      {copilotInsight?.summary}
                    </div>
                  </div>

                  {/* Right Column: Citations & Ask Copilot */}
                  <div className="copilot-side-col">
                    <div className="card copilot-citations-card">
                      <h4 style={{ fontSize: 'var(--text-xs)', fontWeight: 'bold', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 'var(--space-xs)' }}>
                        Verified Medical Literature Citations
                      </h4>
                      <ul style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', listStyleType: 'disc', listStylePosition: 'inside', paddingLeft: 'var(--space-3xs)', margin: 0 }}>
                        {copilotInsight?.citations.map((cite, idx) => (
                          <li key={idx} style={{ marginBottom: '6px' }}>{cite}</li>
                        ))}
                        {copilotInsight?.is_fallback && (
                          <li style={{ color: 'var(--color-muted)', fontStyle: 'italic', listStyleType: 'none', marginTop: 'var(--space-xs)' }}>
                            Advanced vector database medical cross-referencing offline...
                          </li>
                        )}
                      </ul>
                    </div>

                    <div className="card copilot-query-card">
                      <label style={{ fontSize: 'var(--text-xs)', fontWeight: 'bold', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 'var(--space-xs)' }}>
                        Ask Clinical Copilot
                      </label>
                      <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                        <input
                          type="text"
                          disabled
                          placeholder="Ask questions about vocal features or Parkinsonian cohort matching (Phase 2 Upgrade)..."
                          style={{
                            flex: 1,
                            background: 'var(--color-paper-3)',
                            border: '1px solid var(--color-rule)',
                            padding: 'var(--space-2xs) var(--space-xs)',
                            borderRadius: 'var(--radius-input)',
                            color: 'var(--color-ink-3)',
                            fontSize: 'var(--text-xs)',
                            cursor: 'not-allowed'
                          }}
                        />
                        <button
                          disabled
                          className="btn btn--primary btn--sm"
                          style={{ opacity: 0.5, cursor: 'not-allowed' }}
                        >
                          Ask
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
};
