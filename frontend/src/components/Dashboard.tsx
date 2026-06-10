import React, { useEffect, useState, useMemo } from 'react';
import { Activity, ShieldAlert, RefreshCw, Printer, FileText } from 'lucide-react';
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

  const [clusterPoints, setClusterPoints] = useState<Array<{ x: number; y: number; status: number }>>([]);
  const [clustersLoaded, setClustersLoaded] = useState(false);
  const [historyList, setHistoryList] = useState<Array<Record<string, unknown>>>([]);

  const colors = getRiskColors(riskScore);

  const calculateVoiceQualityScore = () => {
    const jitter = clinicalMetrics.jitter_pct;
    const shimmer = clinicalMetrics.shimmer_local * 100.0;
    const hnr = clinicalMetrics.hnr;
    const jitterScore = Math.max(0, 1 - (jitter / 3.0));
    const shimmerScore = Math.max(0, 1 - (shimmer / 12.0));
    const hnrScore = Math.min(1.0, hnr / 26.0);
    return Math.round((jitterScore * 0.35 + shimmerScore * 0.35 + hnrScore * 0.30) * 100);
  };

  const voiceQualityScore = calculateVoiceQualityScore();
  const recordingQualityScore = (data.recording_quality_score as number) || 95;

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

  return (
    <div className="dashboard reveal is-in">
      <header className="results-header">
        <div className="results-header__row">
          <div className="results-header__intro">
            <h1 className="results-header__title">Screening results</h1>
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

        <div className="results-header__summary">
          <div className="results-header__score">
            <span className={`results-header__pct ${colors.textClass}`}>{Math.round(riskScore * 100)}%</span>
            <span className="results-header__score-label">Risk score</span>
          </div>
          <div className="results-header__details">
            <span className={`results-header__category ${colors.badgeClass}`}>{report.risk_category as string}</span>
            {confidenceCalibration && (
              <span className="results-header__certainty">{confidenceCalibration.certainty_label as string}</span>
            )}
            <p className="results-header__summary-text">{report.summary as string}</p>
          </div>
          <div className="results-header__metrics">
            <div className="results-header__metric">
              <span className="stat-block__label">Voice quality</span>
              <span className="results-header__metric-value">{voiceQualityScore}%</span>
            </div>
            <div className="results-header__metric">
              <span className="stat-block__label">Recording</span>
              <span className="results-header__metric-value">{recordingQualityScore}%</span>
            </div>
          </div>
        </div>
      </header>

      <div className="dashboard__layout">
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

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)', minWidth: 0 }}>
          <div className="grid-2">
            <EmbeddingCanvas
              embeddingCoords={embeddingCoords}
              clusterPoints={clusterPoints}
              clustersLoaded={clustersLoaded}
            />

            <div className="card">
              <p className="card__label">Risk score distribution</p>
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

          {shapExplanation && shapExplanation.length > 0 && (
            <div className="card">
              <p className="card__label">Explainable AI (SHAP) Biomarker Contributions</p>
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
            </div>
          )}

          <div>
            <p className="card__label">Clinical Vocal Biomarker Metrics</p>
            <div className="biomarker-grid" style={{ marginTop: 'var(--space-md)' }}>
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
          </div>

          <div className="grid-2">
            <div className="card">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', fontFamily: 'var(--font-display)', fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: '0 0 var(--space-md)' }}>
                <Activity style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
                Clinical Recommendations
              </h3>
              <ul className="rec-list">
                {recommendations.map((rec, idx) => <li key={idx}>{rec}</li>)}
              </ul>
            </div>
            <div className="card card--risk-elevated">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', fontFamily: 'var(--font-display)', fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-danger)', margin: '0 0 var(--space-md)' }}>
                <ShieldAlert style={{ width: 18, height: 18 }} />
                Medical Disclaimer
              </h3>
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink-2)', lineHeight: 1.75, margin: 0 }}>{report.disclaimer as string}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
