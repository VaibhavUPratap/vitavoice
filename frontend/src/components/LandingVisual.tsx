import { useEffect, useRef, useState } from 'react';
import { Activity } from 'lucide-react';

/** Interactive Vocal Signal & Feature Analyzer component — replaces the 3D canvas torus */
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

    // Track mouse coordinates
    const onMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      mouseRef.current = { x, y, clientX: e.clientX, clientY: e.clientY, active: true };

      // Update vertical scanner line and cursor dot position in DOM directly for 60fps performance
      if (scanlineRef.current) {
        scanlineRef.current.setAttribute('x1', `${x}`);
        scanlineRef.current.setAttribute('x2', `${x}`);
        scanlineRef.current.style.opacity = '1';
      }

      // Estimate a point on the main wave path to place the cursor dot
      const width = rect.width;
      const height = rect.height;
      const cy = height * 0.5;
      const normX = x / width;
      const phase = phaseRef.current;

      // Calculate path1's y-coordinate at this X to make the dot stick to the wave
      const baseWave = Math.sin(normX * 8 + phase) * 22;
      const subHarmonic = Math.cos(normX * 16 - phase * 0.7) * 8;
      let waveY = cy + baseWave + subHarmonic;

      // Add mouse proximity warp (local ripple)
      const ripple = Math.sin(normX * 120 + phase * 3.5) * 4;
      waveY += ripple;

      if (cursorDotRef.current) {
        cursorDotRef.current.setAttribute('cx', `${x}`);
        cursorDotRef.current.setAttribute('cy', `${waveY}`);
        cursorDotRef.current.style.opacity = '1';
      }

      // Update tooltip content and position
      if (tooltipRef.current) {
        const estFreq = Math.round(85 + normX * 170); // simulated fundamental frequency (F0)
        const estDb = (-12 - (y / height) * 18).toFixed(1); // simulated decibels
        tooltipRef.current.style.transform = `translate(${x + 12}px, ${y - 48}px)`;
        tooltipRef.current.style.opacity = '1';
        tooltipRef.current.innerHTML = `
          <div class="vocal-tooltip__title">f₀: ${estFreq} Hz</div>
          <div class="vocal-tooltip__meta">amp: ${estDb} dB</div>
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
    // Generate paths for three overlapping wave shapes
    const draw = () => {
      const container = containerRef.current;
      if (!container) return;

      const w = container.clientWidth || 380;
      const h = container.clientHeight || 280;
      const cy = h * 0.5;

      if (!reducedMotion) {
        phaseRef.current += 0.035;
      }

      const phase = phaseRef.current;
      const mouse = mouseRef.current;

      // Build SVG path strings
      let d1 = '';
      let d2 = '';
      let d3 = '';

      const steps = 60;
      for (let i = 0; i <= steps; i++) {
        const normX = i / steps;
        const x = normX * w;

        // Wave 1: Principal wave (electric cobalt)
        let base1 = Math.sin(normX * 8 + phase) * 22;
        let sub1 = Math.cos(normX * 16 - phase * 0.7) * 8;

        // Wave 2: Harmonic wave (teal accent)
        let base2 = Math.sin(normX * 11 - phase * 1.2) * 16;
        let sub2 = Math.cos(normX * 22 + phase * 0.5) * 5;

        // Wave 3: Background noise floor (gray/ink)
        let base3 = Math.cos(normX * 5 + phase * 0.3) * 10;
        let sub3 = Math.sin(normX * 13 + phase * 1.5) * 3;

        // Proximity warp logic: if mouse is active, perturb waves near the mouse X
        if (mouse.active) {
          const mouseNormX = mouse.x / w;
          const dist = Math.abs(normX - mouseNormX);
          const influence = Math.exp(-Math.pow(dist * 7, 2)); // Gaussian envelope

          if (influence > 0.01) {
            // High frequency vocal jitter modulation
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
          <span className="vocal-analyzer__title">vocal.spectral_map_v2</span>
        </div>
        <div className="vocal-analyzer__status">
          <span className="vocal-analyzer__led" />
          <span>Nominal</span>
        </div>
      </div>

      {/* 2. Parameters bar */}
      <div className="vocal-analyzer__meta-bar">
        <span>SR: 16,000 HZ</span>
        <span>CHANNEL: CH_01</span>
        <span>GAIN: AUTO</span>
        <span>F0: STABLE</span>
      </div>

      {/* 3. Waveform display area */}
      <div className="vocal-analyzer__wave-wrap">
        {/* Glow backdrop grid */}
        <div className="vocal-analyzer__grid-lines" />

        {/* Dynamic SVG waves */}
        <svg className="vocal-analyzer__svg" viewBox="0 0 100% 100%" preserveAspectRatio="none">
          <defs>
            <linearGradient id="cobaltGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="oklch(62% 0.15 220)" stopOpacity="0.4" />
              <stop offset="50%" stopColor="var(--color-accent)" stopOpacity="1" />
              <stop offset="100%" stopColor="oklch(58% 0.20 256)" stopOpacity="0.4" />
            </linearGradient>
            <linearGradient id="tealGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="oklch(70% 0.12 170)" stopOpacity="0.2" />
              <stop offset="100%" stopColor="oklch(62% 0.14 200)" stopOpacity="0.8" />
            </linearGradient>
          </defs>

          {/* Background voice envelope range */}
          <rect x="0" y="25%" width="100%" height="50%" fill="oklch(58% 0.20 256 / 0.03)" />

          {/* Paths */}
          <path ref={path3Ref} fill="none" stroke="oklch(52% 0.012 256)" strokeWidth="1" strokeDasharray="3 3" opacity="0.35" />
          <path ref={path2Ref} fill="none" stroke="url(#tealGradient)" strokeWidth="1.5" opacity="0.65" />
          <path ref={path1Ref} fill="none" stroke="url(#cobaltGradient)" strokeWidth="2.5" />

          {/* Interactive cursor lines/dots (DOM-manipulated) */}
          <line ref={scanlineRef} x1="0" y1="0" x2="0" y2="100%" stroke="var(--color-accent)" strokeWidth="1" strokeDasharray="2 2" style={{ opacity: 0, transition: 'opacity 0.2s' }} />
          <circle ref={cursorDotRef} r="5" fill="var(--color-accent)" stroke="var(--color-paper)" strokeWidth="2" style={{ opacity: 0, transition: 'opacity 0.2s' }} />
        </svg>

        {/* Floating tooltip */}
        <div ref={tooltipRef} className="vocal-tooltip" style={{ opacity: 0 }} />

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
          <div className="vocal-mini-card__label">HNR (dB)</div>
          <div className="vocal-mini-card__row">
            <span className="vocal-mini-card__value">26.8 dB</span>
            <span className="vocal-mini-card__badge vocal-mini-card__badge--ok">STABLE</span>
          </div>
        </div>
      </div>
    </div>
  );
}
