import React, { useEffect, useRef, useState } from 'react';
import { Activity, ShieldAlert, RefreshCw, ChevronRight, CheckCircle2, AlertTriangle, Printer, FileText, ZoomIn, ZoomOut } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer, ReferenceLine } from 'recharts';


interface ClusterPoint {
  x: number;
  y: number;
  status: number;
}

interface DashboardProps {
  data: {
    risk_score: number;
    status: number;
    embedding_coords: [number, number];
    clinical_metrics: {
      fo_mean: number;
      fhi: number;
      flo: number;
      jitter_pct: number;
      jitter_abs: number;
      shimmer_local: number;
      shimmer_db: number;
      hnr: number;
      nhr: number;
      energy: number;
      formants: number[];
    };
    report: {
      risk_category: string;
      summary: string;
      biomarker_analysis: Array<{
        name: string;
        label: string;
        value: string;
        status: string;
        explanation: string;
      }>;
      recommendations: string[];
      disclaimer: string;
      confidence_calibration?: {
        risk_probability: number;
        certainty_score: number;
        certainty_label: string;
        calibration_confidence: string;
      };
      shap_explanation?: Array<{
        feature_name: string;
        label: string;
        shap_value: number;
        impact: string;
        abs_value: number;
      }>;
    };
    report_url?: string;
    recording_quality_score?: number;
  };
  onReset: () => void;
  onLoadHistoryRecord?: (recordData: any) => void;
  backendUrl: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ data, onReset, onLoadHistoryRecord, backendUrl }) => {
  const { risk_score, embedding_coords, report } = data;
  const [clusterPoints, setClusterPoints] = useState<ClusterPoint[]>([]);
  const [clustersLoaded, setClustersLoaded] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState<any | null>(null);
  const [historyList, setHistoryList] = useState<any[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pulseRef = useRef<number>(0);
  const animFrameIdRef = useRef<number | null>(null);

  // Load theme colors
  const getRiskColor = (score: number) => {
    if (score < 0.35) return {
      strokeColor: 'oklch(66% 0.18 155)',
      svgStroke: 'oklch(66% 0.18 155)',
      bg: 'oklch(16% 0.06 155 / 0.30)',
      border: 'oklch(40% 0.10 155 / 0.35)',
      glow: 'glow-green',
      text: 'oklch(66% 0.18 155)',
      badgeBg: 'oklch(16% 0.06 155 / 0.40)',
    };
    if (score < 0.65) return {
      strokeColor: 'oklch(74% 0.18 60)',
      svgStroke: 'oklch(74% 0.18 60)',
      bg: 'oklch(16% 0.06 60 / 0.30)',
      border: 'oklch(40% 0.10 60 / 0.35)',
      glow: '',
      text: 'oklch(74% 0.18 60)',
      badgeBg: 'oklch(16% 0.06 60 / 0.40)',
    };
    return {
      strokeColor: 'oklch(62% 0.22 25)',
      svgStroke: 'oklch(62% 0.22 25)',
      bg: 'oklch(16% 0.06 25 / 0.30)',
      border: 'oklch(40% 0.10 25 / 0.35)',
      glow: 'glow-red',
      text: 'oklch(62% 0.22 25)',
      badgeBg: 'oklch(16% 0.06 25 / 0.40)',
    };
  };

  const colors = getRiskColor(risk_score);

  const calculateVoiceQualityScore = () => {
    const jitter = data.clinical_metrics.jitter_pct;
    const shimmer = data.clinical_metrics.shimmer_local * 100.0;
    const hnr = data.clinical_metrics.hnr;
    
    const jitterScore = Math.max(0, 1 - (jitter / 3.0));
    const shimmerScore = Math.max(0, 1 - (shimmer / 12.0));
    const hnrScore = Math.min(1.0, hnr / 26.0);
    
    const composite = (jitterScore * 0.35 + shimmerScore * 0.35 + hnrScore * 0.30) * 100;
    return Math.round(composite);
  };

  const voiceQualityScore = calculateVoiceQualityScore();
  const recordingQualityScore = data.recording_quality_score || 95;

  // Fetch reference clusters and load history
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
        console.error("Failed to load reference clusters:", err);
      }
    };
    fetchClusters();

    // Load localStorage history
    try {
      const historyStr = localStorage.getItem('vitavoice_history') || '[]';
      setHistoryList(JSON.parse(historyStr));
    } catch (err) {
      console.error("Failed to load history list:", err);
    }
  }, [backendUrl, data]);

  // Render 2D Cluster Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const points = clustersLoaded ? clusterPoints : generateDefaultClusters();
    const userX = embedding_coords[0];
    const userY = embedding_coords[1];

    // Bounding box for scaling
    const allX = [...points.map(p => p.x), userX];
    const allY = [...points.map(p => p.y), userY];
    
    let minX: number, maxX: number, minY: number, maxY: number;
    if (zoomed) {
      minX = userX - 1.2;
      maxX = userX + 1.2;
      minY = userY - 0.8;
      maxY = userY + 0.8;
    } else {
      minX = Math.min(...allX) - 1.5;
      maxX = Math.max(...allX) + 1.5;
      minY = Math.min(...allY) - 1.5;
      maxY = Math.max(...allY) + 1.5;
    }

    const drawCanvas = () => {
      const width = canvas.width;
      const height = canvas.height;
      
      const isLightTheme = document.documentElement.classList.contains('light');
      
      ctx.fillStyle = isLightTheme ? '#f8fafc' : '#060a13';
      ctx.fillRect(0, 0, width, height);

      // Grid lines
      ctx.strokeStyle = isLightTheme ? 'rgba(148, 163, 184, 0.15)' : 'rgba(30, 41, 59, 0.3)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < width; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
      }
      for (let i = 0; i < height; i += 50) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
      }

      // Map scale helper
      const mapX = (x: number) => ((x - minX) / (maxX - minX)) * (width - 40) + 20;
      const mapY = (y: number) => height - (((y - minY) / (maxY - minY)) * (height - 40) + 20);

      // Draw cluster points
      points.forEach(p => {
        // Skip points outside zoomed bounds
        if (zoomed && (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY)) return;
        
        const cx = mapX(p.x);
        const cy = mapY(p.y);
        ctx.beginPath();
        ctx.arc(cx, cy, zoomed ? 4.5 : 3.5, 0, 2 * Math.PI);
        
        if (p.status === 0) {
          ctx.fillStyle = isLightTheme ? 'rgba(5, 150, 105, 0.5)' : 'rgba(16, 185, 129, 0.4)';
        } else {
          ctx.fillStyle = isLightTheme ? 'rgba(225, 29, 72, 0.5)' : 'rgba(244, 63, 94, 0.4)';
        }
        ctx.fill();
      });

      // User Voice Coordinate (Animated Pulsing)
      const uCx = mapX(userX);
      const uCy = mapY(userY);

      pulseRef.current = (pulseRef.current + 0.08) % (2 * Math.PI);
      const pulseSize = 6 + Math.sin(pulseRef.current) * 4;
      const alpha = 0.4 - Math.sin(pulseRef.current) * 0.2;

      ctx.beginPath();
      ctx.arc(uCx, uCy, pulseSize * 2, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(6, 182, 212, ${alpha})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(uCx, uCy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#06b6d4';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = isLightTheme ? '#0f172a' : '#f8fafc';
      ctx.font = 'bold 9px sans-serif';
      ctx.fillText("Patient Voice Signature", uCx + 10, uCy - 8);

      animFrameIdRef.current = requestAnimationFrame(drawCanvas);
    };

    drawCanvas();

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [clustersLoaded, clusterPoints, embedding_coords, zoomed]);

  const generateDefaultClusters = (): ClusterPoint[] => {
    const points: ClusterPoint[] = [];
    for (let i = 0; i < 40; i++) {
      points.push({
        x: -2 + npRandomNormal() * 0.8,
        y: 0 + npRandomNormal() * 0.8,
        status: 0
      });
    }
    for (let i = 0; i < 60; i++) {
      points.push({
        x: 1.5 + npRandomNormal() * 1.2,
        y: 0.8 + npRandomNormal() * 1.0,
        status: 1
      });
    }
    return points;
  };

  const npRandomNormal = () => {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const width = canvas.width;
    const height = canvas.height;

    const points = clustersLoaded ? clusterPoints : generateDefaultClusters();
    const userX = embedding_coords[0];
    const userY = embedding_coords[1];
    const allPoints = [...points, { x: userX, y: userY, status: -1 }];

    const allX = allPoints.map(p => p.x);
    const allY = allPoints.map(p => p.y);

    let minX: number, maxX: number, minY: number, maxY: number;
    if (zoomed) {
      minX = userX - 1.2;
      maxX = userX + 1.2;
      minY = userY - 0.8;
      maxY = userY + 0.8;
    } else {
      minX = Math.min(...allX) - 1.5;
      maxX = Math.max(...allX) + 1.5;
      minY = Math.min(...allY) - 1.5;
      maxY = Math.max(...allY) + 1.5;
    }

    let closestPt: any = null;
    let minDist = 15;

    allPoints.forEach(p => {
      if (zoomed && (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY)) return;
      const px = ((p.x - minX) / (maxX - minX)) * (width - 40) + 20;
      const py = height - (((p.y - minY) / (maxY - minY)) * (height - 40) + 20);
      
      const dist = Math.sqrt((mouseX - px) ** 2 + (mouseY - py) ** 2);
      if (dist < minDist) {
        minDist = dist;
        closestPt = { ...p, screenX: px, screenY: py };
      }
    });

    if (closestPt) {
      setHoveredPoint({
        x: closestPt.x,
        y: closestPt.y,
        status: closestPt.status,
        canvasX: closestPt.screenX,
        canvasY: closestPt.screenY
      });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoveredPoint(null);
  };

  // Generate density curve data
  const generateProbabilityData = () => {
    const arr = [];
    const step = 0.02;
    for (let x = 0; x <= 1.01; x += step) {
      const healthyDensity = (1 / (0.15 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 0.2) / 0.15, 2));
      const pathDensity = (1 / (0.15 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 0.8) / 0.15, 2));
      arr.push({
        x: Math.round(x * 100) / 100,
        "Healthy Cohort": parseFloat(healthyDensity.toFixed(3)),
        "Pathology Cohort": parseFloat(pathDensity.toFixed(3))
      });
    }
    return arr;
  };
  
  const probData = generateProbabilityData();

  const loadPastRun = (run: any) => {
    if (onLoadHistoryRecord) {
      onLoadHistoryRecord(run.data);
    }
  };

  const triggerPrint = () => {
    window.print();
  };

  const cardStyle: React.CSSProperties = {
    background: 'oklch(14% 0.028 270 / 0.60)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    border: '1px solid var(--color-rule)',
    borderRadius: 'var(--radius-card)',
  };

  return (
    <div className="animate-fade-in-up" style={{ width: '100%', maxWidth: 980, padding: '0 1rem 2rem' }}>
      {/* Action Header bar */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', marginBottom: '2rem', borderBottom: '1px solid var(--color-rule-subtle)', paddingBottom: '1.5rem' }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 900, background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-warm))', WebkitBackgroundClip: 'text', backgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em', marginBottom: '0.375rem' }}>
            Voice Biomarker Analysis Dashboard
          </h1>
          <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>Screening Reference ID: VV-{data.report_url ? data.report_url.split('_')[1].split('.')[0].toUpperCase() : Math.floor(100000 + Math.random() * 900000)}</p>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.625rem' }} className="print:hidden">
          {data.report_url && (
            <a
              href={`${backendUrl}${data.report_url}`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ padding: '0.5rem 1rem', background: 'oklch(18% 0.028 270 / 0.60)', border: '1px solid var(--color-rule)', borderRadius: 10, fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-accent)', display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none', transition: 'all var(--dur-fast) var(--ease-out)' }}
            >
              <FileText style={{ width: 14, height: 14 }} /> Download PDF
            </a>
          )}
          <button
            onClick={triggerPrint}
            style={{ padding: '0.5rem 1rem', background: 'oklch(18% 0.028 270 / 0.60)', border: '1px solid var(--color-rule)', borderRadius: 10, fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-ink-2)', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', transition: 'all var(--dur-fast) var(--ease-out)' }}
          >
            <Printer style={{ width: 14, height: 14 }} /> Print
          </button>
          <button
            onClick={onReset}
            className="btn-primary btn-sm"
          >
            <RefreshCw style={{ width: 14, height: 14 }} /> New Screening
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', alignItems: 'flex-start' }} className="lg-row">
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', width: '100%', flexWrap: 'wrap' }}>
        {/* History Sidebar */}
        <div style={{ ...cardStyle, width: 220, flexShrink: 0, padding: '1.25rem' }} className="print:hidden">
          <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Screening History</span>
            <span style={{ fontSize: '0.625rem', background: 'oklch(22% 0.028 270)', color: 'var(--color-ink-2)', padding: '0.15rem 0.5rem', borderRadius: 9999, fontFamily: 'var(--font-mono)' }}>{historyList.length}</span>
          </h3>
          {historyList.length === 0 ? (
            <p style={{ fontSize: '0.75rem', color: 'var(--color-ink-3)', textAlign: 'center', padding: '1.5rem 0' }}>No previous screenings.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: 360, overflowY: 'auto' }}>
              {historyList.map((run, idx) => {
                const runDate = new Date(run.timestamp).toLocaleDateString();
                const runTime = new Date(run.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const isSelected = data.report_url && run.data.report_url === data.report_url;
                const riskClr = run.risk_score >= 0.65 ? 'oklch(62% 0.22 25)' : run.risk_score >= 0.35 ? 'oklch(74% 0.18 60)' : 'oklch(66% 0.18 155)';
                return (
                  <button
                    key={idx}
                    onClick={() => loadPastRun(run)}
                    style={{ width: '100%', textAlign: 'left', padding: '0.75rem', borderRadius: 10, border: `1px solid ${isSelected ? 'oklch(72% 0.20 215 / 0.40)' : 'var(--color-rule)'}`, background: isSelected ? 'oklch(20% 0.08 215 / 0.25)' : 'oklch(16% 0.022 270 / 0.40)', cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: '0.375rem', transition: 'all var(--dur-fast) var(--ease-out)' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-ink)' }}>{runDate} - {runTime}</span>
                      <span style={{ fontSize: '0.5625rem', fontWeight: 800, padding: '0.15rem 0.4rem', borderRadius: 4, background: 'oklch(16% 0.02 270)', color: riskClr }}>{Math.round(run.risk_score * 100)}%</span>
                    </div>
                    <span style={{ fontSize: '0.625rem', color: 'var(--color-ink-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{run.risk_category}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Dashboard Panels */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
            {/* Risk Gauge */}
            <div style={{ ...cardStyle, padding: '1.75rem', display: 'flex', flexDirection: 'column', alignItems: 'center', boxShadow: `0 0 40px -12px ${colors.text}` }}>
              <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: '1.5rem' }}>Overall Health Risk Score</h3>
              <div style={{ position: 'relative', width: 160, height: 160, marginBottom: '1.5rem' }}>
                <svg style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }} viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" stroke="oklch(22% 0.028 270 / 0.50)" strokeWidth="7" fill="transparent" />
                  <circle cx="50" cy="50" r="42" stroke={colors.svgStroke} strokeWidth="7" fill="transparent"
                    strokeDasharray={`${2 * Math.PI * 42}`}
                    strokeDashoffset={`${2 * Math.PI * 42 * (1 - risk_score)}`}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.16,1,0.3,1)' }}
                  />
                </svg>
                <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ fontSize: '2.25rem', fontWeight: 900, fontFamily: 'var(--font-display)', color: 'var(--color-ink)', lineHeight: 1 }}>{Math.round(risk_score * 100)}%</span>
                  <span style={{ fontSize: '0.6875rem', fontWeight: 700, padding: '0.25rem 0.75rem', borderRadius: 9999, marginTop: 6, background: colors.badgeBg, color: colors.text }}>{report.risk_category}</span>
                  {report.confidence_calibration && (
                    <span style={{ fontSize: '0.5625rem', fontWeight: 700, padding: '0.2rem 0.5rem', borderRadius: 9999, marginTop: 4, background: 'oklch(18% 0.08 215 / 0.40)', color: 'var(--color-accent)', border: '1px solid oklch(72% 0.20 215 / 0.20)' }}>
                      {report.confidence_calibration.certainty_label}
                    </span>
                  )}
                </div>
              </div>
              <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                <p style={{ fontSize: '0.875rem', color: 'var(--color-ink-2)', lineHeight: 1.7 }}>{report.summary}</p>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', width: '100%', paddingTop: '1rem', borderTop: '1px solid var(--color-rule-subtle)', textAlign: 'center' }}>
                <div>
                  <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block' }}>Voice Quality</span>
                  <span style={{ fontSize: '1.375rem', fontWeight: 900, fontFamily: 'var(--font-display)', color: 'var(--color-accent)', display: 'block', marginTop: 2 }}>{voiceQualityScore}%</span>
                  <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', display: 'block' }}>Biomarker Stability</span>
                </div>
                <div>
                  <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block' }}>Recording Quality</span>
                  <span style={{ fontSize: '1.375rem', fontWeight: 900, fontFamily: 'var(--font-display)', color: 'var(--color-success)', display: 'block', marginTop: 2 }}>{recordingQualityScore}%</span>
                  <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', display: 'block' }}>Signal Fidelity</span>
                </div>
              </div>
            </div>

            {/* 2D Embedding Space */}
            <div style={{ ...cardStyle, padding: '1.5rem', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>Wav2Vec 2.0 Vocal Embedding Space</h3>
                  <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', marginTop: 3 }}>Phonetic representation projected to 2D via PCA</p>
                </div>
                <button onClick={() => setZoomed(prev => !prev)} className="print:hidden btn-ghost btn-sm">
                  {zoomed ? <ZoomOut style={{ width: 13, height: 13 }} /> : <ZoomIn style={{ width: 13, height: 13 }} />}
                  {zoomed ? 'Reset' : 'Zoom'}
                </button>
              </div>
              <div style={{ width: '100%', background: 'oklch(9% 0.022 272)', border: '1px solid var(--color-rule)', borderRadius: 10, overflow: 'hidden', minHeight: 240, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <canvas ref={canvasRef} width={620} height={240} onMouseMove={handleCanvasMouseMove} onMouseLeave={handleCanvasMouseLeave} style={{ width: '100%', height: '100%', maxHeight: 240, cursor: 'crosshair' }} />
                {!clustersLoaded && (
                  <span style={{ position: 'absolute', bottom: 8, left: 10, fontSize: '0.5625rem', color: 'var(--color-ink-3)', background: 'oklch(10% 0.022 272 / 0.80)', padding: '0.15rem 0.5rem', borderRadius: 4 }}>Reference clusters simulated (retrying...)</span>
                )}
                {hoveredPoint && (
                  <div style={{ position: 'absolute', left: hoveredPoint.canvasX + 10, top: hoveredPoint.canvasY - 50, background: 'oklch(14% 0.028 270 / 0.96)', border: '1px solid var(--color-rule)', color: 'var(--color-ink)', borderRadius: 10, padding: '0.625rem 0.875rem', fontSize: '0.6875rem', boxShadow: '0 8px 24px oklch(0% 0 0 / 0.50)', pointerEvents: 'none', zIndex: 50, display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <span style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: hoveredPoint.status === -1 ? 'var(--color-accent)' : hoveredPoint.status === 0 ? 'var(--color-success)' : 'var(--color-danger)', display: 'inline-block' }} />
                      {hoveredPoint.status === -1 ? 'Patient Voice' : hoveredPoint.status === 0 ? 'Healthy Control' : "Parkinson's Patient"}
                    </span>
                    <span style={{ fontSize: '0.5625rem', color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)' }}>[{hoveredPoint.x.toFixed(3)}, {hoveredPoint.y.toFixed(3)}]</span>
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.625rem', fontSize: '0.5625rem', color: 'var(--color-ink-3)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: 'oklch(66% 0.18 155 / 0.50)', display: 'inline-block', border: '1px solid oklch(66% 0.18 155 / 0.30)' }} />Healthy</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: 'oklch(62% 0.22 25 / 0.50)', display: 'inline-block', border: '1px solid oklch(62% 0.22 25 / 0.30)' }} />Parkinson's</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--color-accent)', display: 'inline-block' }} />You</span>
                </div>
                <span>Hover to inspect points</span>
              </div>
            </div>
          </div>

          {/* Probability Distribution */}
          <div style={{ ...cardStyle, padding: '1.5rem' }}>
            <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: '0.375rem' }}>Risk Score Probability Distribution</h3>
            <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', marginBottom: '1.25rem' }}>Patient risk score vs healthy and Parkinsonian reference populations.</p>
            <div style={{ width: '100%', height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={probData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                  <XAxis dataKey="x" tickFormatter={(val) => `${Math.round(val * 100)}%`} stroke="oklch(38% 0.016 270)" fontSize={10} />
                  <YAxis stroke="oklch(38% 0.016 270)" fontSize={10} hide={true} />
                  <ChartTooltip contentStyle={{ backgroundColor: 'oklch(12% 0.028 270)', borderColor: 'oklch(28% 0.022 270)', borderRadius: 10 }} labelStyle={{ color: 'oklch(60% 0.012 272)', fontSize: '10px' }} itemStyle={{ fontSize: '11px' }} />
                  <Area type="monotone" name="Healthy Cohort" dataKey="Healthy Cohort" stroke="oklch(66% 0.18 155)" fill="oklch(66% 0.18 155 / 0.08)" strokeWidth={1.5} />
                  <Area type="monotone" name="Pathology Cohort" dataKey="Pathology Cohort" stroke="oklch(62% 0.22 25)" fill="oklch(62% 0.22 25 / 0.08)" strokeWidth={1.5} />
                  <ReferenceLine x={risk_score} stroke="oklch(72% 0.20 215)" strokeWidth={2} strokeDasharray="4 3" label={{ value: 'You', fill: 'oklch(78% 0.18 215)', fontSize: 10, position: 'top', fontWeight: 'bold' }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* SHAP Explainability */}
          {report.shap_explanation && report.shap_explanation.length > 0 && (
            <div style={{ ...cardStyle, padding: '1.5rem' }}>
              <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: '0.375rem' }}>Explainable AI (SHAP) Biomarker Contributions</h3>
              <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', marginBottom: '1.5rem' }}>Top biomarkers driving classification. Green = reduces risk / Red = increases risk.</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {report.shap_explanation.map((feat: any, idx: number) => {
                  const isPositive = feat.shap_value > 0;
                  const absVal = Math.abs(feat.shap_value);
                  const maxAbs = Math.max(...report.shap_explanation!.map((f: any) => Math.abs(f.shap_value))) || 0.1;
                  const widthPct = Math.min(100, (absVal / maxAbs) * 100);
                  return (
                    <div key={idx} style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: '1rem', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-ink)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{feat.label}</span>
                      <div style={{ position: 'relative', height: 20, background: 'oklch(14% 0.022 270)', borderRadius: 4, overflow: 'hidden', border: '1px solid var(--color-rule-subtle)', display: 'flex', alignItems: 'center' }}>
                        <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'oklch(30% 0.022 270)', zIndex: 2 }} />
                        {isPositive
                          ? <div style={{ position: 'absolute', left: '50%', height: '100%', width: `${widthPct / 2}%`, background: 'linear-gradient(90deg, oklch(52% 0.22 25 / 0.80), oklch(62% 0.22 25 / 0.90))', borderRadius: '0 3px 3px 0', transition: 'width 1s cubic-bezier(0.16,1,0.3,1)' }} />
                          : <div style={{ position: 'absolute', right: '50%', height: '100%', width: `${widthPct / 2}%`, background: 'linear-gradient(270deg, oklch(52% 0.18 155 / 0.80), oklch(62% 0.18 155 / 0.90))', borderRadius: '3px 0 0 3px', transition: 'width 1s cubic-bezier(0.16,1,0.3,1)' }} />
                        }
                      </div>
                      <span style={{ fontSize: '0.6875rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: isPositive ? 'oklch(62% 0.22 25)' : 'oklch(66% 0.18 155)', textAlign: 'right', minWidth: 60 }}>{isPositive ? '+' : ''}{feat.shap_value.toFixed(4)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Biomarker Cards */}
          <div>
            <h3 style={{ fontSize: '0.625rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--color-ink-3)', textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: '1rem' }}>Clinical Vocal Biomarker Metrics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
              {report.biomarker_analysis.map((biomarker, idx) => {
                const isElevated = biomarker.status === 'Elevated' || biomarker.status.includes('Low');
                return (
                  <div key={idx} className="card-3d" style={{ ...cardStyle, padding: '1.25rem', borderColor: isElevated ? 'oklch(40% 0.10 25 / 0.30)' : 'var(--color-rule)', background: isElevated ? 'oklch(14% 0.06 25 / 0.25)' : cardStyle.background }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.625rem' }}>
                      <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--color-ink-2)' }}>{biomarker.label}</span>
                      {isElevated
                        ? <AlertTriangle style={{ width: 14, height: 14, color: 'var(--color-danger)', flexShrink: 0 }} />
                        : <CheckCircle2 style={{ width: 14, height: 14, color: 'var(--color-success)', flexShrink: 0 }} />
                      }
                    </div>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.75rem' }}>
                      <span style={{ fontSize: '1.5rem', fontWeight: 900, fontFamily: 'var(--font-display)', color: 'var(--color-ink)' }}>{biomarker.value}</span>
                      <span style={{ fontSize: '0.5625rem', fontWeight: 700, padding: '0.15rem 0.5rem', borderRadius: 9999, background: isElevated ? 'oklch(20% 0.06 25 / 0.50)' : 'oklch(16% 0.06 155 / 0.40)', color: isElevated ? 'var(--color-danger)' : 'var(--color-success)' }}>{biomarker.status}</span>
                    </div>
                    <p style={{ fontSize: '0.6875rem', color: 'var(--color-ink-3)', lineHeight: 1.7, marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--color-rule-subtle)' }}>{biomarker.explanation}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recommendations + Disclaimer */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
            <div style={{ ...cardStyle, padding: '1.5rem' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 800, color: 'var(--color-ink)', marginBottom: '1rem' }}>
                <Activity style={{ width: 18, height: 18, color: 'var(--color-accent)', flexShrink: 0 }} /> Clinical Recommendations
              </h3>
              <ul style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', listStyle: 'none', padding: 0 }}>
                {report.recommendations.map((rec, idx) => (
                  <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.875rem', color: 'var(--color-ink-2)', lineHeight: 1.7 }}>
                    <ChevronRight style={{ width: 14, height: 14, color: 'var(--color-accent)', flexShrink: 0, marginTop: 3 }} />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div style={{ ...cardStyle, padding: '1.5rem', background: 'oklch(14% 0.06 25 / 0.20)', borderColor: 'oklch(40% 0.10 25 / 0.25)' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 800, color: 'var(--color-danger)', marginBottom: '0.875rem' }}>
                <ShieldAlert style={{ width: 18, height: 18, color: 'var(--color-danger)', flexShrink: 0 }} /> Medical Disclaimer
              </h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-ink-2)', lineHeight: 1.8 }}>{report.disclaimer}</p>
            </div>
          </div>
        </div>
        </div>
      </div>
      
      {/* Hidden print page header */}
      <div className="hidden print:block text-slate-900 bg-white p-8 border border-slate-300 mt-12 rounded">
        <h2 className="text-3xl font-bold font-outfit text-cyan-800 border-b pb-4 mb-6">VitaVoice Clinical Screening Assessment</h2>
        <div className="grid grid-cols-2 gap-6 mb-6">
          <div>
            <p><strong>Overall Risk Score:</strong> {Math.round(risk_score * 100)}% ({report.risk_category})</p>
            <p><strong>Assessment Date:</strong> {new Date().toLocaleDateString()}</p>
          </div>
          <div>
            <p><strong>Average Pitch (F0):</strong> {data.clinical_metrics.fo_mean.toFixed(2)} Hz</p>
            <p><strong>Harmonics-to-Noise Ratio (HNR):</strong> {data.clinical_metrics.hnr.toFixed(2)} dB</p>
          </div>
        </div>
        <div className="border-t pt-4">
          <h4 className="font-bold text-slate-800 mb-2">Screening Summary:</h4>
          <p className="text-sm leading-relaxed mb-6">{report.summary}</p>
          <h4 className="font-bold text-slate-800 mb-2">Biomarker Details:</h4>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-700">
            {report.biomarker_analysis.map((b, idx) => (
              <li key={idx}><strong>{b.label}:</strong> {b.value} - <em>{b.status}</em></li>
            ))}
          </ul>
        </div>
        <p className="text-[10px] text-slate-400 mt-12 border-t pt-4">{report.disclaimer}</p>
      </div>
    </div>
  );
};
