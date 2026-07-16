import { useEffect, useRef, useState } from 'react';
import { Activity } from 'lucide-react';

/** Interactive Vocal Signal & Feature Analyzer component — styled as a Lumen Apparatus */
export function LandingVisual() {
  const containerRef = useRef<HTMLDivElement>(null);
  const path1Ref = useRef<SVGPathElement>(null);
  const path2Ref = useRef<SVGPathElement>(null);
  const path3Ref = useRef<SVGPathElement>(null);
  const scanlineRef = useRef<SVGLineElement>(null);
  const cursorDotRef = useRef<SVGCircleElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const phaseRef = useRef(0);
  const frameRef = useRef(0);
  const mouseRef = useRef({ x: 0, y: 0, clientX: 0, clientY: 0, active: false });

  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    setReducedMotion(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      mouseRef.current = { x, y, clientX: e.clientX, clientY: e.clientY, active: true };

      if (scanlineRef.current) {
        scanlineRef.current.setAttribute('x1', `${x}`);
        scanlineRef.current.setAttribute('x2', `${x}`);
        scanlineRef.current.style.opacity = '0.5';
      }

      const width = rect.width;
      const height = rect.height;
      const cy = height * 0.5;
      const normX = x / width;
      const phase = phaseRef.current;

      const baseWave = Math.sin(normX * 8 + phase) * 22;
      const subHarmonic = Math.cos(normX * 16 - phase * 0.7) * 8;
      let waveY = cy + baseWave + subHarmonic;

      const ripple = Math.sin(normX * 120 + phase * 3.5) * 4;
      waveY += ripple;

      if (cursorDotRef.current) {
        cursorDotRef.current.setAttribute('cx', `${x}`);
        cursorDotRef.current.setAttribute('cy', `${waveY}`);
        cursorDotRef.current.style.opacity = '1';
      }

      if (tooltipRef.current) {
        const estFreq = Math.round(85 + normX * 170);
        const estDb = (-12 - (y / height) * 18).toFixed(1);
        tooltipRef.current.style.transform = `translate(${x + 12}px, ${y - 48}px)`;
        tooltipRef.current.style.opacity = '1';
        tooltipRef.current.innerHTML = `
          <div class="vocal-tooltip__title">F0: ${estFreq} HZ</div>
          <div class="vocal-tooltip__meta">AMP: ${estDb} DB</div>
        `;
      }
    };

    const onMouseLeave = () => {
      mouseRef.current.active = false;
      if (scanlineRef.current) scanlineRef.current.style.opacity = '0';
      if (cursorDotRef.current) cursorDotRef.current.style.opacity = '0';
      if (tooltipRef.current) tooltipRef.current.style.opacity = '0';
    };

    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('mouseleave', onMouseLeave);

    return () => {
      container.removeEventListener('mousemove', onMouseMove);
      container.removeEventListener('mouseleave', onMouseLeave);
    };
  }, []);

  useEffect(() => {
    const draw = () => {
      const container = containerRef.current;
      if (!container) return;

      const w = container.clientWidth || 380;
      const h = container.clientHeight || 280;
      const cy = h * 0.5;

      if (!reducedMotion) {
        phaseRef.current += 0.025;
      }

      const phase = phaseRef.current;
      const mouse = mouseRef.current;

      let d1 = '';
      let d2 = '';
      let d3 = '';

      const steps = 60;
      for (let i = 0; i <= steps; i++) {
        const normX = i / steps;
        const x = normX * w;

        let base1 = Math.sin(normX * 8 + phase) * 22;
        let sub1 = Math.cos(normX * 16 - phase * 0.7) * 8;

        let base2 = Math.sin(normX * 11 - phase * 1.2) * 16;
        let sub2 = Math.cos(normX * 22 + phase * 0.5) * 5;

        let base3 = Math.cos(normX * 5 + phase * 0.3) * 10;
        let sub3 = Math.sin(normX * 13 + phase * 1.5) * 3;

        if (mouse.active) {
          const mouseNormX = mouse.x / w;
          const dist = Math.abs(normX - mouseNormX);
          const influence = Math.exp(-Math.pow(dist * 7, 2));

          if (influence > 0.01) {
            const ripple = Math.sin(normX * 120 + phase * 3.5) * 4;
            base1 += ripple * influence;
            base2 += ripple * 0.7 * influence;
            base3 += ripple * 1.5 * influence;
          }
        }

        const y1 = cy + base1 + sub1;
        const y2 = cy + base2 + sub2;
        const y3 = cy + base3 + sub3;

        if (i === 0) {
          d1 = `M ${x} ${y1}`;
          d2 = `M ${x} ${y2}`;
          d3 = `M ${x} ${y3}`;
        } else {
          d1 += ` L ${x} ${y1}`;
          d2 += ` L ${x} ${y2}`;
          d3 += ` L ${x} ${y3}`;
        }
      }

      if (path1Ref.current) path1Ref.current.setAttribute('d', d1);
      if (path2Ref.current) path2Ref.current.setAttribute('d', d2);
      if (path3Ref.current) path3Ref.current.setAttribute('d', d3);

      frameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(frameRef.current);
  }, [reducedMotion]);

  return (
    <div ref={containerRef} className="vocal-analyzer" aria-hidden="true">
      {/* 1. Terminal Header */}
      <div className="vocal-analyzer__header">
        <div className="vocal-analyzer__title-row">
          <Activity className="vocal-analyzer__icon" />
          <span className="vocal-analyzer__title">VOCAL·SPECTRAL·MAP·V2</span>
        </div>
        <div className="vocal-analyzer__status">
          <span className="vocal-analyzer__led" />
          <span>NOMINAL</span>
        </div>
      </div>

      {/* 2. Parameters bar */}
      <div className="vocal-analyzer__meta-bar">
        <span>SR: 16,000 HZ</span>
        <span>CHANNEL: CH_01</span>
        <span>GAIN: CALIBRATED</span>
        <span>F0: LOCK</span>
      </div>

      {/* 3. Waveform display area */}
      <div className="vocal-analyzer__wave-wrap">
        {/* Glow backdrop grid */}
        <div className="vocal-analyzer__grid-lines" />

        {/* Dynamic SVG waves */}
        <svg className="vocal-analyzer__svg" viewBox="0 0 100% 100%" preserveAspectRatio="none">
          <defs>
            <linearGradient id="brassGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="oklch(62% 0.12 50)" stopOpacity="0.3" />
              <stop offset="50%" stopColor="var(--color-accent)" stopOpacity="1" />
              <stop offset="100%" stopColor="oklch(76% 0.17 50)" stopOpacity="0.3" />
            </linearGradient>
            <linearGradient id="coralGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="oklch(58% 0.14 18)" stopOpacity="0.2" />
              <stop offset="100%" stopColor="var(--color-accent-2)" stopOpacity="0.75" />
            </linearGradient>
          </defs>

          {/* Background voice envelope range */}
          <rect x="0" y="25%" width="100%" height="50%" fill="oklch(76% 0.17 50 / 0.02)" />

          {/* Paths */}
          <path ref={path3Ref} fill="none" stroke="oklch(62% 0.010 262)" strokeWidth="1" strokeDasharray="3 3" opacity="0.25" />
          <path ref={path2Ref} fill="none" stroke="url(#coralGradient)" strokeWidth="1.5" opacity="0.5" />
          <path ref={path1Ref} fill="none" stroke="url(#brassGradient)" strokeWidth="2.2" />

          {/* Interactive cursor lines/dots (DOM-manipulated) */}
          <line ref={scanlineRef} x1="0" y1="0" x2="0" y2="100%" stroke="var(--color-accent-2)" strokeWidth="1" strokeDasharray="2 2" style={{ opacity: 0, transition: 'opacity 0.2s' }} />
          <circle ref={cursorDotRef} r="4" fill="var(--color-accent-2)" stroke="var(--color-paper)" strokeWidth="1.5" style={{ opacity: 0, transition: 'opacity 0.2s' }} />
        </svg>

        {/* Floating tooltip */}
        <div ref={tooltipRef} className="vocal-tooltip" style={{ opacity: 0 }} />

        {/* Horizontal leader-line callouts (Lumen signature) */}
        <ul className="callouts">
          <li className="callout callout--left" style={{ top: '22%' }}>
            <span>JITTER · 0.38%</span>
          </li>
          <li className="callout callout--left" style={{ top: '78%' }}>
            <span>SHIMMER · 2.42%</span>
          </li>
          <li className="callout callout--right" style={{ top: '35%' }}>
            <span>HNR · 26.8 DB</span>
          </li>
          <li className="callout callout--right" style={{ top: '65%' }}>
            <span>WAVLM · 768 DIM</span>
          </li>
        </ul>

        {/* Mini 2D PCA Cluster Map on top-right */}
        <div className="vocal-mini-pca">
          <span className="vocal-mini-pca__title">PCA EMBD</span>
          <div className="vocal-mini-pca__points">
            {/* Cluster dots */}
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--healthy" style={{ top: '35%', left: '25%' }} />
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--healthy" style={{ top: '55%', left: '35%' }} />
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--healthy" style={{ top: '25%', left: '45%' }} />
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--pathology" style={{ top: '65%', left: '65%' }} />
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--pathology" style={{ top: '45%', left: '75%' }} />
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--pathology" style={{ top: '75%', left: '80%' }} />
            {/* Blinking user dot */}
            <span className="vocal-mini-pca__dot vocal-mini-pca__dot--you" style={{ top: '42%', left: '52%' }} />
          </div>
        </div>
      </div>

      {/* 4. Bottom feature mini-grid */}
      <div className="vocal-analyzer__metrics">
        <div className="vocal-mini-card">
          <div className="vocal-mini-card__label">JITTER (LOCAL)</div>
          <div className="vocal-mini-card__row">
            <span className="vocal-mini-card__value">0.38%</span>
            <span className="vocal-mini-card__badge vocal-mini-card__badge--ok">NOMINAL</span>
          </div>
        </div>
        <div className="vocal-mini-card">
          <div className="vocal-mini-card__label">SHIMMER (LOCAL)</div>
          <div className="vocal-mini-card__row">
            <span className="vocal-mini-card__value">2.42%</span>
            <span className="vocal-mini-card__badge vocal-mini-card__badge--ok">NOMINAL</span>
          </div>
        </div>
        <div className="vocal-mini-card">
          <div className="vocal-mini-card__label">HNR (DB)</div>
          <div className="vocal-mini-card__row">
            <span className="vocal-mini-card__value">26.8 DB</span>
            <span className="vocal-mini-card__badge vocal-mini-card__badge--ok">STABLE</span>
          </div>
        </div>
      </div>
    </div>
  );
}
