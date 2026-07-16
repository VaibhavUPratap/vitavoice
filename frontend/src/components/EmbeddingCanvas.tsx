import { useEffect, useRef, useState, useCallback } from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';

interface ClusterPoint {
  x: number;
  y: number;
  status: number;
}

interface EmbeddingCanvasProps {
  embeddingCoords: [number, number];
  clusterPoints: ClusterPoint[];
  clustersLoaded: boolean;
}

type HoverInfo = {
  x: number;
  y: number;
  status: number;
  displayX: number;
  displayY: number;
};

function generateFallbackClusters(): ClusterPoint[] {
  const normal = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };
  const points: ClusterPoint[] = [];
  for (let i = 0; i < 40; i++) {
    points.push({ x: -2 + normal() * 0.8, y: normal() * 0.8, status: 0 });
  }
  for (let i = 0; i < 60; i++) {
    points.push({ x: 1.5 + normal() * 1.2, y: 0.8 + normal() * 1.0, status: 1 });
  }
  return points;
}

export function EmbeddingCanvas({ embeddingCoords, clusterPoints, clustersLoaded }: EmbeddingCanvasProps) {
  const [zoomed, setZoomed] = useState(false);
  const [hovered, setHovered] = useState<HoverInfo | null>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 620, h: 240 });

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pulseRef = useRef(0);
  const frameRef = useRef<number>(0);
  const fallbackRef = useRef<ClusterPoint[]>(generateFallbackClusters());

  const [userX, userY] = embeddingCoords;
  const coordsValid = Number.isFinite(userX) && Number.isFinite(userY);
  const points = clustersLoaded && clusterPoints.length > 0 ? clusterPoints : fallbackRef.current;

  const getBounds = useCallback(() => {
    const allX = [...points.map((p) => p.x), coordsValid ? userX : 0];
    const allY = [...points.map((p) => p.y), coordsValid ? userY : 0];
    if (zoomed && coordsValid) {
      return { minX: userX - 1.2, maxX: userX + 1.2, minY: userY - 0.8, maxY: userY + 0.8 };
    }
    return {
      minX: Math.min(...allX) - 0.8,
      maxX: Math.max(...allX) + 0.8,
      minY: Math.min(...allY) - 0.8,
      maxY: Math.max(...allY) + 0.8,
    };
  }, [points, userX, userY, zoomed, coordsValid]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = wrap.clientWidth;
      const h = 240;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      setCanvasSize({ w, h });
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = canvas.width / wrap.clientWidth;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { minX, maxX, minY, maxY } = getBounds();
    const pad = 24;

    const mapX = (x: number, width: number) => pad + ((x - minX) / (maxX - minX)) * (width - pad * 2);
    const mapY = (y: number, height: number) => height - pad - ((y - minY) / (maxY - minY)) * (height - pad * 2);

    const draw = () => {
      const width = wrap.clientWidth;
      const height = 240;

      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-graphite').trim();
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'oklch(38% 0.012 260 / 0.45)';
      ctx.lineWidth = 1;
      for (let gx = pad; gx < width - pad; gx += 40) {
        ctx.beginPath();
        ctx.moveTo(gx, pad);
        ctx.lineTo(gx, height - pad);
        ctx.stroke();
      }
      for (let gy = pad; gy < height - pad; gy += 40) {
        ctx.beginPath();
        ctx.moveTo(pad, gy);
        ctx.lineTo(width - pad, gy);
        ctx.stroke();
      }

      for (const p of points) {
        if (zoomed && (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY)) continue;
        ctx.beginPath();
        ctx.arc(mapX(p.x, width), mapY(p.y, height), zoomed ? 4 : 3, 0, Math.PI * 2);
        ctx.fillStyle = p.status === 0
          ? 'oklch(52% 0.14 155 / 0.55)'
          : 'oklch(52% 0.18 25 / 0.55)';
        ctx.fill();
      }

      if (coordsValid) {
        const uCx = mapX(userX, width);
        const uCy = mapY(userY, height);
        pulseRef.current = (pulseRef.current + 0.04) % (2 * Math.PI);
        const time = pulseRef.current;

        // Draw crosshair scanning lines
        ctx.strokeStyle = 'oklch(58% 0.20 256 / 0.45)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        
        // Horizontal scan line
        ctx.beginPath();
        ctx.moveTo(pad, uCy);
        ctx.lineTo(width - pad, uCy);
        ctx.stroke();

        // Vertical scan line
        ctx.beginPath();
        ctx.moveTo(uCx, pad);
        ctx.lineTo(uCx, height - pad);
        ctx.stroke();

        // Reset line dash
        ctx.setLineDash([]);

        // Scanning ring expanding outward
        const scanRingRadius = ((time * 25) % 40) + 10;
        const scanRingAlpha = 1 - (scanRingRadius - 10) / 40;
        ctx.beginPath();
        ctx.arc(uCx, uCy, scanRingRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `oklch(58% 0.20 256 / ${scanRingAlpha * 0.6})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // High-tech target corner brackets around "Your Voice"
        const boxSize = 10;
        ctx.strokeStyle = 'oklch(58% 0.20 256)';
        ctx.lineWidth = 1.5;
        
        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(uCx - boxSize, uCy - boxSize + 3);
        ctx.lineTo(uCx - boxSize, uCy - boxSize);
        ctx.lineTo(uCx - boxSize + 3, uCy - boxSize);
        ctx.stroke();

        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(uCx + boxSize - 3, uCy - boxSize);
        ctx.lineTo(uCx + boxSize, uCy - boxSize);
        ctx.lineTo(uCx + boxSize, uCy - boxSize + 3);
        ctx.stroke();

        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(uCx - boxSize, uCy + boxSize - 3);
        ctx.lineTo(uCx - boxSize, uCy + boxSize);
        ctx.lineTo(uCx - boxSize + 3, uCy + boxSize);
        ctx.stroke();

        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(uCx + boxSize - 3, uCy + boxSize);
        ctx.lineTo(uCx + boxSize, uCy + boxSize);
        ctx.lineTo(uCx + boxSize, uCy + boxSize - 3);
        ctx.stroke();

        // Center dot
        ctx.beginPath();
        ctx.arc(uCx, uCy, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'oklch(58% 0.20 256)';
        ctx.fill();

        // Inner solid core ring
        ctx.beginPath();
        ctx.arc(uCx, uCy, 5, 0, Math.PI * 2);
        ctx.strokeStyle = 'oklch(96% 0.005 250)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Display coordinate overlay
        ctx.fillStyle = 'oklch(58% 0.20 256)';
        ctx.font = '700 8px "JetBrains Mono", monospace';
        const coordText = `f_lat: [${userX.toFixed(3)}, ${userY.toFixed(3)}]`;
        ctx.fillText(coordText, uCx + 12, uCy + 14);

        // Label
        ctx.fillStyle = 'oklch(96% 0.005 250)';
        ctx.font = '600 10px "Space Grotesk", sans-serif';
        const label = 'YOUR VOICE';
        const labelW = ctx.measureText(label).width;
        let lx = uCx + 12;
        let ly = uCy - 8;
        if (lx + labelW > width - pad) lx = uCx - labelW - 12;
        if (ly < pad) ly = uCy + 22;
        ctx.fillText(label, lx, ly);
      }

      frameRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, [points, userX, userY, zoomed, coordsValid, getBounds, canvasSize.w]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const rect = canvas.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = displayX * scaleX;
    const mouseY = displayY * scaleY;

    const width = wrap.clientWidth;
    const height = 240;
    const dpr = canvas.width / width;
    const logicalX = mouseX / dpr;
    const logicalY = mouseY / dpr;

    const { minX, maxX, minY, maxY } = getBounds();
    const pad = 24;
    const mapX = (x: number) => pad + ((x - minX) / (maxX - minX)) * (width - pad * 2);
    const mapY = (y: number) => height - pad - ((y - minY) / (maxY - minY)) * (height - pad * 2);

    const allPoints = coordsValid
      ? [...points, { x: userX, y: userY, status: -1 }]
      : points;

    let closest: HoverInfo | null = null;
    let minDist = 18;

    for (const p of allPoints) {
      if (zoomed && (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY)) continue;
      const px = mapX(p.x);
      const py = mapY(p.y);
      const dist = Math.hypot(logicalX - px, logicalY - py);
      if (dist < minDist) {
        minDist = dist;
        closest = { x: p.x, y: p.y, status: p.status, displayX, displayY };
      }
    }

    setHovered(closest);
  };

  return (
    <div className="card embedding-card">
      <div className="embedding-card__head">
        <div>
          <p className="card__label" style={{ marginBottom: 'var(--space-2xs)' }}>Deep Speech Latent Topology Space</p>
          <p className="embedding-card__sub">WavLM Neural Embedding Cluster Map</p>
        </div>
        <button
          type="button"
          onClick={() => setZoomed((z) => !z)}
          className="btn btn--ghost btn--sm no-print"
          aria-pressed={zoomed}
        >
          {zoomed ? <ZoomOut style={{ width: 13, height: 13 }} /> : <ZoomIn style={{ width: 13, height: 13 }} />}
          {zoomed ? 'Reset view' : 'Zoom in'}
        </button>
      </div>

      <div ref={wrapRef} className="canvas-wrap">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
        />
        {!clustersLoaded && (
          <span className="canvas-wrap__hint">Using simulated reference clusters</span>
        )}
        {hovered && (
          <div
            className="canvas-wrap__tooltip"
            style={{ left: hovered.displayX + 12, top: hovered.displayY - 48 }}
          >
            <strong>
              {hovered.status === -1 ? 'Your voice' : hovered.status === 0 ? 'Healthy control' : "Parkinson's patient"}
            </strong>
            <span>[{hovered.x.toFixed(3)}, {hovered.y.toFixed(3)}]</span>
          </div>
        )}
      </div>

      <div className="embedding-legend">
        <span><i className="embedding-legend__dot embedding-legend__dot--healthy" />Healthy</span>
        <span><i className="embedding-legend__dot embedding-legend__dot--pathology" />Parkinson&apos;s</span>
        <span><i className="embedding-legend__dot embedding-legend__dot--you" />You</span>
      </div>
    </div>
  );
}
