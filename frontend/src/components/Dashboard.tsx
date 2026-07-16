import React, { useEffect, useState, useMemo } from 'react';
import {
  Activity, ShieldAlert, RefreshCw, Printer, FileText,
  Stethoscope, Brain, AlertTriangle, Mic, BarChart3,
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
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

/* ─── Confidence Ring SVG Component ─────────────────────────── */

function ConfidenceRing({ score, label, reliability }: { score: number; label: string; reliability: string }) {
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const ringColor =
    score >= 90 ? 'var(--color-success)' :
    score >= 70 ? 'var(--color-accent)' :
    score >= 55 ? 'var(--color-warning)' :
    'var(--color-danger)';

  const reliabilityClass =
    reliability === 'High' ? 'bg-risk-low' :
    reliability === 'Moderate' ? 'bg-risk-mid' :
    'bg-risk-high';

  return (
    <div className="confidence-ring">
      <div className="confidence-ring__svg-wrap">
        <svg className="confidence-ring__svg" viewBox="0 0 100 100">
          <circle className="confidence-ring__bg" cx="50" cy="50" r={radius} />
          <circle
            className="confidence-ring__fill"
            cx="50"
            cy="50"
            r={radius}
            stroke={ringColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="confidence-ring__value">
          <span className="confidence-ring__pct">{Math.round(score)}%</span>
          <span className="confidence-ring__label">Confidence</span>
        </div>
      </div>
      <span className="stat-block__label">Confidence: {label}</span>
      <span className={`confidence-ring__reliability ${reliabilityClass}`}>
        Reliability: {reliability}
      </span>
    </div>
  );
}

export const Dashboard: React.FC<DashboardProps> = ({ data, onReset, onLoadHistoryRecord, backendUrl }) => {
  const riskScore = data.risk_score as number;
  const embeddingCoords = data.embedding_coords as [number, number];
  const report = data.report as Record<string, unknown>;
  const clinicalMetrics = data.clinical_metrics as Record<string, number>;
  const biomarkerAnalysis = report.biomarker_analysis as Array<Record<string, string>>;
  const recommendations = report.recommendations as string[];
  const shapExplanation = report.shap_explanation as Array<Record<string, unknown>> | undefined;
  const confidenceCalibration = report.confidence_calibration as Record<string, unknown> | undefined;
  const reportUrl = data.report_url as string | undefined;

  // New enriched data
  const recordingQuality = data.recording_quality as Record<string, unknown> | undefined;
  const confidenceScore = (data.confidence_score as number) || 50;
  const confidenceLabel = (data.confidence_label as string) || 'N/A';
  const predictionReliability = (data.prediction_reliability as string) || 'N/A';
  const topBiomarkers = (data.top_biomarkers as Array<Record<string, unknown>>) || [];
  const naturalLanguageExplanation = (data.natural_language_explanation as string) || '';
  const recommendationText = (data.recommendation as string) || '';
  const responsibleAiPoints = (data.responsible_ai_points as string[]) || [
    'This application is intended for preliminary screening only.',
    'It is not a medical diagnosis.',
    'Voice recordings are processed only for analysis.',
    'Long-term storage of recordings is disabled.',
    'Consult a qualified neurologist for clinical diagnosis.',
  ];
  const biomarkerStatuses = (data.biomarker_statuses as Array<Record<string, unknown>>) || [];

  const [clusterPoints, setClusterPoints] = useState<Array<{ x: number; y: number; status: number }>>([]);
  const [clustersLoaded, setClustersLoaded] = useState(false);
  const [historyList, setHistoryList] = useState<Array<Record<string, unknown>>>([]);

  const colors = getRiskColors(riskScore);

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

  // Quality metrics
  const qualityStars = (recordingQuality?.quality_stars as string) || '★★★★★';
  const qualityWarning = (recordingQuality?.quality_warning as string) || null;

  return (
    <div className="dashboard reveal is-in">
      {/* ─── Header ─────────────────────────────────────────── */}
      <header className="results-header">
        <div className="results-header__row">
          <div className="results-header__intro">
            <h1 className="results-header__title">Clinical Screening Summary</h1>
            <p className="results-header__meta">{refId} · {screenedDate}</p>
          </div>
          <div className="results-header__actions no-print">
            {reportUrl && (
              <a href={`${backendUrl}${reportUrl}`} target="_blank" rel="noopener noreferrer" className="btn btn--outline btn--sm">
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
            Screening History
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

          {/* ═══ Section 1: Clinical Screening Summary ══════ */}
          <div className="dashboard-section clinical-summary">
            <div className="clinical-summary__risk">
              <span className="clinical-summary__risk-label">Estimated Risk</span>
              <span className={`clinical-summary__risk-pct ${colors.textClass}`}>
                {Math.round(riskScore * 100)}%
              </span>
              <span className={`clinical-summary__risk-cat ${colors.badgeClass}`}>
                {report.risk_category as string}
              </span>
            </div>

            <div className="clinical-summary__divider" />

            <div className="clinical-summary__details">
              <ConfidenceRing
                score={confidenceScore}
                label={confidenceLabel}
                reliability={predictionReliability}
              />
            </div>
          </div>

          {/* ═══ Section 2: Recording Quality ═══════════════ */}
          {recordingQuality && (
            <div className="dashboard-section recording-quality">
              <div className="recording-quality__header">
                <div>
                  <p className="section-label">
                    <Mic className="section-label__icon" />
                    Recording Quality
                  </p>
                </div>
                <span className="recording-quality__stars">{qualityStars}</span>
              </div>
              <div className="recording-quality__grid">
                <div className="rq-metric">
                  <span className="rq-metric__label">Duration</span>
                  <span className="rq-metric__value">{recordingQuality.duration_seconds as number}s</span>
                </div>
                <div className="rq-metric">
                  <span className="rq-metric__label">Background Noise</span>
                  <span className="rq-metric__value">{recordingQuality.background_noise_pct as number}%</span>
                </div>
                <div className="rq-metric">
                  <span className="rq-metric__label">Speech Coverage</span>
                  <span className="rq-metric__value">{recordingQuality.speech_coverage_pct as number}%</span>
                </div>
                <div className="rq-metric">
                  <span className="rq-metric__label">Silence</span>
                  <span className="rq-metric__value">{recordingQuality.silence_ratio_pct as number}%</span>
                </div>
                <div className="rq-metric">
                  <span className="rq-metric__label">Microphone Status</span>
                  <span className="rq-metric__value">{recordingQuality.mic_status as string}</span>
                </div>
                <div className="rq-metric">
                  <span className="rq-metric__label">Suitable for Analysis</span>
                  <span className="rq-metric__value" style={{ color: (recordingQuality.suitable_for_analysis as boolean) ? 'var(--color-success)' : 'var(--color-danger)' }}>
                    {(recordingQuality.suitable_for_analysis as boolean) ? 'Yes' : 'No'}
                  </span>
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

          {/* ═══ Section 3: Biomarker Analysis ══════════════ */}
          <div className="dashboard-section">
            <p className="section-label">
              <BarChart3 className="section-label__icon" />
              Biomarker Analysis
            </p>
            {biomarkerStatuses.length > 0 ? (
              <div className="biomarker-status-grid">
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
            ) : (
              /* Fallback to legacy biomarker cards */
              <div className="biomarker-grid">
                {biomarkerAnalysis.map((biomarker, idx) => {
                  const isElevated = biomarker.status === 'Elevated' || biomarker.status.includes('Low');
                  return (
                    <div key={idx} className={`biomarker-card${isElevated ? ' biomarker-card--elevated' : ''}`}>
                      <div className="biomarker-card__header">
                        <span className="biomarker-card__name">{biomarker.label}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-2xs)' }}>
                        <span className="biomarker-card__value">{biomarker.value}</span>
                        <span className={`biomarker-card__status ${isElevated ? 'bg-risk-high' : 'bg-risk-low'}`}>{biomarker.status}</span>
                      </div>
                      <p className="biomarker-card__explain">{biomarker.explanation}</p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ═══ Section 4: Voice Projection Map + Distribution ══ */}
          <div className="dashboard-section grid-2">
            <EmbeddingCanvas
              embeddingCoords={embeddingCoords}
              clusterPoints={clusterPoints}
              clustersLoaded={clustersLoaded}
            />

            <div className="card">
              <p className="card__label">Risk Score Distribution</p>
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-3)', marginBottom: 'var(--space-md)' }}>
                Your score vs healthy and Parkinsonian reference cohorts.
              </p>
              <div style={{ width: '100%', height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={probData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                    <XAxis dataKey="x" tickFormatter={(val) => `${Math.round(val * 100)}%`} stroke="var(--color-ink-3)" fontSize={10} />
                    <YAxis stroke="var(--color-ink-3)" fontSize={10} hide />
                    <ChartTooltip contentStyle={{ backgroundColor: 'var(--color-paper)', borderColor: 'var(--color-rule)', borderRadius: 'var(--radius-btn)' }} />
                    <Area type="monotone" name="Healthy Cohort" dataKey="Healthy Cohort" stroke="var(--color-success)" fill="oklch(52% 0.14 155 / 0.08)" strokeWidth={1.5} />
                    <Area type="monotone" name="Pathology Cohort" dataKey="Pathology Cohort" stroke="var(--color-danger)" fill="oklch(52% 0.18 25 / 0.08)" strokeWidth={1.5} />
                    <ReferenceLine x={riskScore} stroke="var(--color-accent)" strokeWidth={2} strokeDasharray="4 3" label={{ value: 'You', fill: 'var(--color-accent)', fontSize: 10, position: 'top', fontWeight: 'bold' }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* ═══ Section 5: Explainable AI ══════════════════ */}
          <div className="dashboard-section xai-panel">
            <div className="xai-panel__header">
              <Brain style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
              <h3 className="xai-panel__title">Explainable AI</h3>
            </div>

            {/* Directional biomarker indicators */}
            {topBiomarkers.length > 0 ? (
              <>
                <p className="section-label" style={{ marginBottom: 'var(--space-sm)' }}>Top Contributing Biomarkers</p>
                <div className="xai-biomarker-list">
                  {topBiomarkers.map((bm, idx) => {
                    const direction = bm.direction as string;
                    const descriptor = bm.descriptor as string;
                    const shapValue = bm.shap_value as number;
                    const isPositive = shapValue > 0;
                    return (
                      <div key={idx} className="xai-biomarker">
                        <span className={`xai-arrow ${isPositive ? 'xai-arrow--up' : 'xai-arrow--down'}`}>
                          {direction}
                        </span>
                        <span className="xai-biomarker__text">{descriptor}</span>
                        <span className={`xai-biomarker__shap ${isPositive ? 'text-risk-high' : 'text-risk-low'}`}>
                          {isPositive ? '+' : ''}{shapValue.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : shapExplanation && shapExplanation.length > 0 ? (
              /* Fallback to legacy SHAP bars */
              <>
                <p className="card__label">SHAP Biomarker Contributions</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', marginTop: 'var(--space-md)' }}>
                  {shapExplanation.map((feat, idx) => {
                    const shapValue = feat.shap_value as number;
                    const isPositive = shapValue > 0;
                    const maxAbs = Math.max(...shapExplanation.map((f) => Math.abs(f.shap_value as number))) || 0.1;
                    const widthPct = Math.min(100, (Math.abs(shapValue) / maxAbs) * 100);
                    return (
                      <div key={idx} className="shap-row">
                        <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-ink)' }}>{feat.label as string}</span>
                        <div className="shap-bar">
                          <span className="shap-bar__center" />
                          {isPositive ? (
                            <span className="shap-bar__pos" style={{ width: `${widthPct / 2}%` }} />
                          ) : (
                            <span className="shap-bar__neg" style={{ width: `${widthPct / 2}%` }} />
                          )}
                        </div>
                        <span className={isPositive ? 'text-risk-high' : 'text-risk-low'} style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', fontWeight: 600 }}>
                          {isPositive ? '+' : ''}{shapValue.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : null}

            {/* Natural language explanation */}
            {naturalLanguageExplanation && (
              <>
                <p className="section-label" style={{ marginTop: topBiomarkers.length > 0 ? 0 : 'var(--space-md)' }}>Model Explanation</p>
                <div className="xai-explanation">
                  {naturalLanguageExplanation}
                </div>
              </>
            )}

            {/* Legacy SHAP bar chart (still shown below XAI if legacy data is available and new XAI used top biomarkers) */}
            {topBiomarkers.length > 0 && shapExplanation && shapExplanation.length > 0 && (
              <div style={{ marginTop: 'var(--space-lg)' }}>
                <p className="section-label">SHAP Feature Importance</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                  {shapExplanation.map((feat, idx) => {
                    const shapValue = feat.shap_value as number;
                    const isPositive = shapValue > 0;
                    const maxAbs = Math.max(...shapExplanation.map((f) => Math.abs(f.shap_value as number))) || 0.1;
                    const widthPct = Math.min(100, (Math.abs(shapValue) / maxAbs) * 100);
                    return (
                      <div key={idx} className="shap-row">
                        <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-ink)' }}>{feat.label as string}</span>
                        <div className="shap-bar">
                          <span className="shap-bar__center" />
                          {isPositive ? (
                            <span className="shap-bar__pos" style={{ width: `${widthPct / 2}%` }} />
                          ) : (
                            <span className="shap-bar__neg" style={{ width: `${widthPct / 2}%` }} />
                          )}
                        </div>
                        <span className={isPositive ? 'text-risk-high' : 'text-risk-low'} style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', fontWeight: 600 }}>
                          {isPositive ? '+' : ''}{shapValue.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* ═══ Section 6: Recommendation & Responsible AI ═ */}
          <div className="dashboard-section grid-2">
            {/* Recommendation */}
            <div className="recommendation-card">
              <div className="recommendation-card__header">
                <Stethoscope style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
                <h3 className="recommendation-card__title">Clinical Recommendation</h3>
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
            <div className="responsible-ai">
              <div className="responsible-ai__header">
                <ShieldAlert style={{ width: 18, height: 18, color: 'var(--color-danger)' }} />
                <h3 className="responsible-ai__title">Responsible AI</h3>
              </div>
              <ul className="responsible-ai__list">
                {responsibleAiPoints.map((point, idx) => (
                  <li key={idx}>{point}</li>
                ))}
              </ul>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};
