import { useEffect, useRef } from 'react';

/** Rotating 3D voice-wave torus — canvas projection, no external 3D lib. */
export function LandingVisual() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<number>(0);
  const mouseRef = useRef({ x: 0, y: 0 });
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const onMove = (e: MouseEvent) => {
      const rect = wrap.getBoundingClientRect();
      mouseRef.current = {
        x: (e.clientX - rect.left) / rect.width - 0.5,
        y: (e.clientY - rect.top) / rect.height - 0.5,
      };
    };
    wrap.addEventListener('mousemove', onMove);

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const N = 220;
    const pts: { u: number; v: number }[] = [];
    for (let i = 0; i < N; i++) {
      pts.push({ u: (i / N) * Math.PI * 2, v: ((i * 7) % N / N) * Math.PI * 2 });
    }

    const draw = () => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      if (!reducedMotion) timeRef.current += 0.012;

      const styles = getComputedStyle(document.documentElement);
      const accent = styles.getPropertyValue('--color-accent').trim();
      const paper = styles.getPropertyValue('--color-paper-2').trim();
      const rule = styles.getPropertyValue('--color-rule').trim();

      ctx.fillStyle = paper;
      ctx.fillRect(0, 0, w, h);

      const cx = w * 0.5 + mouseRef.current.x * 18;
      const cy = h * 0.5 + mouseRef.current.y * 14;
      const t = timeRef.current;
      const R = Math.min(w, h) * 0.28;
      const r = R * 0.38;

      const projected: { x: number; y: number; z: number; i: number }[] = [];

      for (let i = 0; i < pts.length; i++) {
        const { u, v } = pts[i];
        const wave = 1 + 0.12 * Math.sin(u * 5 + t * 2.4) * Math.cos(v * 3 + t * 1.6);
        let x = (R + r * Math.cos(v)) * Math.cos(u) * wave;
        let y = (R + r * Math.cos(v)) * Math.sin(u) * wave;
        let z = r * Math.sin(v) * wave;

        const cosY = Math.cos(t * 0.7);
        const sinY = Math.sin(t * 0.7);
        const cosX = Math.cos(t * 0.45);
        const sinX = Math.sin(t * 0.45);

        const x1 = x * cosY + z * sinY;
        const z1 = -x * sinY + z * cosY;
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;

        const persp = 1 / (3.2 - z2 / R);
        projected.push({
          x: cx + x1 * persp,
          y: cy + y2 * persp,
          z: z2,
          i,
        });
      }

      projected.sort((a, b) => a.z - b.z);

      for (const p of projected) {
        const depth = (p.z + R) / (R * 2);
        const alpha = 0.25 + depth * 0.65;
        const radius = 1.2 + depth * 2.2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `oklch(58% 0.20 256 / ${alpha.toFixed(2)})`;
        ctx.fill();
      }

      ctx.beginPath();
      for (let i = 0; i <= 64; i++) {
        const u = (i / 64) * Math.PI * 2;
        const wave = 1 + 0.18 * Math.sin(u * 8 + t * 3);
        let x = R * Math.cos(u) * wave;
        let y = R * Math.sin(u) * wave * 0.35;
        let z = 0.25 * R * Math.sin(u * 4 + t * 2);

        const cosY = Math.cos(t * 0.7);
        const sinY = Math.sin(t * 0.7);
        const x1 = x * cosY + z * sinY;
        const z1 = -x * sinY + z * cosY;
        const persp = 1 / (3.2 - z1 / R);
        const px = cx + x1 * persp;
        const py = cy + y * persp;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;

      ctx.strokeStyle = rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(cx, cy + R * 0.55, R * 0.9, R * 0.22, 0, 0, Math.PI * 2);
      ctx.globalAlpha = 0.35;
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (!reducedMotion) frameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(frameRef.current);
      ro.disconnect();
      wrap.removeEventListener('mousemove', onMove);
    };
  }, []);

  return (
    <div ref={wrapRef} className="landing-visual" aria-hidden="true">
      <canvas ref={canvasRef} />
      <div className="landing-visual__caption">
        <span className="landing-visual__dot" />
        Live voice embedding topology
      </div>
    </div>
  );
}
