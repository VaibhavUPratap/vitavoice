/**
 * Handwriting.jsx — Handwriting Parkinson's Screening Module
 *
 * Standalone page: accepts spiral + wave drawing uploads,
 * posts to /predict/handwriting, and displays the result.
 * Does not import or modify any existing component.
 */

import { useState, useRef, useCallback } from 'react';

const BACKEND_URL = import.meta.env.VITE_API_URL || '';

// ─── Score gauge colours ──────────────────────────────────────────────────────
function riskColour(score) {
  if (score < 0.3) return 'var(--color-success)';
  if (score < 0.6) return 'var(--color-warning)';
  return 'var(--color-danger)';
}

function riskLabel(score) {
  if (score < 0.3) return 'Low risk';
  if (score < 0.6) return 'Moderate risk';
  return 'High risk';
}

// ─── Single drop-zone component ───────────────────────────────────────────────
function DrawingUploader({ id, label, hint, file, onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)', flex: 1 }}>
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--text-xs)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          color: 'var(--color-accent)',
        }}
      >
        {label}
      </span>
      <div
        id={id}
        role="button"
        tabIndex={0}
        aria-label={`Upload ${label}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        style={{
          border: `1.5px dashed ${dragging ? 'var(--color-accent)' : 'var(--color-rule-2)'}`,
          borderRadius: 'var(--radius-card)',
          background: dragging
            ? 'oklch(76% 0.17 50 / 0.06)'
            : 'var(--color-paper-2)',
          minHeight: 200,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          transition: 'all var(--dur-short) var(--ease-out)',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {preview ? (
          <img
            src={preview}
            alt={`${label} preview`}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              maxHeight: 220,
              padding: '0.5rem',
            }}
          />
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 'var(--space-xs)',
              padding: 'var(--space-md)',
              textAlign: 'center',
            }}
          >
            {/* Upload icon */}
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <rect width="40" height="40" rx="10" fill="oklch(76% 0.17 50 / 0.10)" />
              <path
                d="M20 27V18M20 18l-4 4M20 18l4 4"
                stroke="var(--color-accent)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M13 23v3a2 2 0 002 2h10a2 2 0 002-2v-3"
                stroke="var(--color-accent)"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
            <p
              style={{
                fontSize: 'var(--text-sm)',
                color: 'var(--color-ink-2)',
                margin: 0,
              }}
            >
              drag &amp; drop or{' '}
              <span style={{ color: 'var(--color-accent)', textDecoration: 'underline' }}>
                browse
              </span>
            </p>
            <p
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-ink-3)',
                margin: 0,
                fontFamily: 'var(--font-mono)',
              }}
            >
              {hint}
            </p>
          </div>
        )}
      </div>
      {file && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--space-xs)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-3)',
              fontFamily: 'var(--font-mono)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: '70%',
            }}
          >
            {file.name}
          </span>
          <button
            onClick={(e) => { e.stopPropagation(); onFile(null); }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-ink-3)',
              cursor: 'pointer',
              fontSize: 'var(--text-xs)',
              fontFamily: 'var(--font-mono)',
              padding: '2px 6px',
              borderRadius: 'var(--radius-btn)',
            }}
          >
            ✕ remove
          </button>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => onFile(e.target.files[0] || null)}
      />
    </div>
  );
}

// ─── Arc gauge ────────────────────────────────────────────────────────────────
function ArcGauge({ score }) {
  const pct    = Math.min(1, Math.max(0, score));
  const radius = 70;
  const cx = 100, cy = 100;
  const startAngle = Math.PI * 0.75;               // 135°
  const sweepAngle = Math.PI * 1.5;                // 270° arc
  const endAngle  = startAngle + sweepAngle * pct;

  const polarToXY = (angle, r) => ({
    x: cx + r * Math.cos(angle),
    y: cy + r * Math.sin(angle),
  });

  const bgStart = polarToXY(startAngle, radius);
  const bgEnd   = polarToXY(startAngle + sweepAngle, radius);
  const fgEnd   = polarToXY(endAngle, radius);

  const bgPath = [
    `M ${bgStart.x} ${bgStart.y}`,
    `A ${radius} ${radius} 0 1 1 ${bgEnd.x} ${bgEnd.y}`,
  ].join(' ');

  const largeArc = sweepAngle * pct > Math.PI ? 1 : 0;
  const fgPath = [
    `M ${bgStart.x} ${bgStart.y}`,
    `A ${radius} ${radius} 0 ${largeArc} 1 ${fgEnd.x} ${fgEnd.y}`,
  ].join(' ');

  const colour = riskColour(score);

  return (
    <svg
      viewBox="0 0 200 160"
      width="200"
      height="160"
      aria-label={`Risk gauge: ${Math.round(score * 100)}%`}
    >
      {/* Background arc */}
      <path
        d={bgPath}
        fill="none"
        stroke="var(--color-rule-2)"
        strokeWidth="14"
        strokeLinecap="round"
      />
      {/* Filled arc */}
      {pct > 0 && (
        <path
          d={fgPath}
          fill="none"
          stroke={colour}
          strokeWidth="14"
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${colour})` }}
        />
      )}
      {/* Score text */}
      <text
        x={cx}
        y={cy + 8}
        textAnchor="middle"
        fontSize="28"
        fontFamily="var(--font-mono)"
        fontWeight="700"
        fill={colour}
      >
        {Math.round(pct * 100)}%
      </text>
      <text
        x={cx}
        y={cy + 26}
        textAnchor="middle"
        fontSize="10"
        fontFamily="var(--font-mono)"
        fill="var(--color-ink-3)"
        letterSpacing="0.08em"
      >
        PD PROBABILITY
      </text>
    </svg>
  );
}

// ─── Main page component ──────────────────────────────────────────────────────
export function Handwriting({ onBack }) {
  const [spiralFile, setSpiralFile] = useState(null);
  const [waveFile,   setWaveFile]   = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);   // { handwriting_score: float }
  const [error,      setError]      = useState(null);

  const canSubmit = spiralFile && waveFile && !loading;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    fd.append('spiral_file', spiralFile);
    fd.append('wave_file',   waveFile);

    try {
      const res = await fetch(`${BACKEND_URL}/predict/handwriting`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        if (res.status === 503) {
          throw new Error('Models are still training — please wait a few minutes and try again.');
        }
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSpiralFile(null);
    setWaveFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div
      id="handwriting-page"
      style={{
        width: '100%',
        maxWidth: 820,
        margin: '0 auto',
        padding: 'var(--space-lg) var(--page-gutter)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-lg)',
      }}
    >
      {/* ── Page header ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-md)' }}>
        <button
          id="hw-back-btn"
          onClick={onBack}
          style={{
            background: 'none',
            border: '1px solid var(--color-rule-2)',
            borderRadius: 'var(--radius-btn)',
            color: 'var(--color-ink-3)',
            cursor: 'pointer',
            padding: '6px 12px',
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--text-xs)',
            transition: 'all var(--dur-short) var(--ease-out)',
            flexShrink: 0,
            marginTop: 4,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-accent)';
            e.currentTarget.style.color = 'var(--color-accent)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-rule-2)';
            e.currentTarget.style.color = 'var(--color-ink-3)';
          }}
        >
          ← back
        </button>

        <div>
          <p
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-accent)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              margin: 0,
            }}
          >
            Handwriting Analysis Module
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <img src="/logo.svg" alt="VitaVoice" style={{ width: 32, height: 32, borderRadius: 6 }} />
            <h1
              style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'var(--text-xl)',
                color: 'var(--color-ink)',
                margin: '0.2em 0 0.4em',
                fontWeight: 400,
              }}
            >
              drawing biomarker screening
            </h1>
          </div>
          <p style={{ color: 'var(--color-ink-3)', fontSize: 'var(--text-sm)', margin: 0 }}>
            upload a spiral drawing and a wave drawing to run the ResNet18-based Parkinson's screening pipeline.
          </p>
        </div>
      </div>

      {/* ── Info strip ───────────────────────────────────────────────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 'var(--space-sm)',
        }}
      >
        {[
          { label: 'Spiral model', value: 'ResNet18' },
          { label: 'Wave model',   value: 'ResNet18' },
          { label: 'Fusion',       value: 'Logistic Reg.' },
        ].map(({ label, value }) => (
          <div
            key={label}
            style={{
              background: 'var(--color-paper-2)',
              border: '1px solid var(--color-rule)',
              borderRadius: 'var(--radius-card)',
              padding: 'var(--space-sm)',
              textAlign: 'center',
            }}
          >
            <p
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-xs)',
                color: 'var(--color-ink-3)',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                margin: '0 0 4px',
              }}
            >
              {label}
            </p>
            <p
              style={{
                fontSize: 'var(--text-sm)',
                color: 'var(--color-ink)',
                margin: 0,
                fontWeight: 500,
              }}
            >
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* ── Upload panel ─────────────────────────────────────────────── */}
      {!result && (
        <div
          style={{
            background: 'var(--color-paper-2)',
            border: '1px solid var(--color-rule)',
            borderRadius: 'var(--radius-card)',
            padding: 'var(--space-lg)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-md)',
          }}
        >
          <div
            style={{
              display: 'flex',
              gap: 'var(--space-md)',
              flexWrap: 'wrap',
            }}
          >
            <DrawingUploader
              id="spiral-uploader"
              label="Spiral drawing"
              hint="Archimedes / concentric spiral"
              file={spiralFile}
              onFile={setSpiralFile}
            />
            <DrawingUploader
              id="wave-uploader"
              label="Wave drawing"
              hint="Repeating wave / meander pattern"
              file={waveFile}
              onFile={setWaveFile}
            />
          </div>

          {error && (
            <div
              style={{
                background: 'var(--color-danger-bg)',
                border: '1px solid var(--color-danger)',
                borderRadius: 'var(--radius-card)',
                padding: 'var(--space-sm) var(--space-md)',
                color: 'var(--color-danger)',
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-xs)',
              }}
            >
              ⚠ error: {error}
            </div>
          )}

          <button
            id="hw-submit-btn"
            disabled={!canSubmit}
            onClick={handleSubmit}
            style={{
              background: canSubmit
                ? 'var(--color-accent)'
                : 'var(--color-paper-3)',
              color: canSubmit
                ? 'var(--color-accent-ink)'
                : 'var(--color-ink-3)',
              border: 'none',
              borderRadius: 'var(--radius-btn)',
              padding: '12px 28px',
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-sm)',
              fontWeight: 600,
              cursor: canSubmit ? 'pointer' : 'not-allowed',
              letterSpacing: '0.05em',
              transition: 'all var(--dur-short) var(--ease-out)',
              alignSelf: 'flex-end',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              opacity: canSubmit ? 1 : 0.5,
            }}
          >
            {loading ? (
              <>
                <span
                  style={{
                    display: 'inline-block',
                    width: 14,
                    height: 14,
                    border: '2px solid currentColor',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 0.7s linear infinite',
                  }}
                />
                analysing...
              </>
            ) : (
              'Run Screening'
            )}
          </button>
        </div>
      )}

      {/* ── Results panel ────────────────────────────────────────────── */}
      {result && (
        <div
          id="hw-results-panel"
          style={{
            background: 'var(--color-paper-2)',
            border: `1px solid ${riskColour(result.handwriting_score)}`,
            borderRadius: 'var(--radius-card)',
            padding: 'var(--space-lg)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 'var(--space-md)',
            boxShadow: `0 0 32px ${riskColour(result.handwriting_score)}22`,
          }}
        >
          <p
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-3)',
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              margin: 0,
            }}
          >
            Handwriting Screening Result
          </p>

          <ArcGauge score={result.handwriting_score} />

          <div style={{ textAlign: 'center' }}>
            <p
              style={{
                fontSize: 'var(--text-md)',
                fontWeight: 600,
                color: riskColour(result.handwriting_score),
                margin: '0 0 0.4em',
              }}
            >
              {riskLabel(result.handwriting_score)}
            </p>
            <p
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-ink-3)',
                fontFamily: 'var(--font-mono)',
                margin: 0,
              }}
            >
              combined score: {result.handwriting_score.toFixed(4)} ·{' '}
              {result.handwriting_score >= 0.5
                ? 'elevated pd biomarker signature detected'
                : 'no significant pd biomarker signature detected'}
            </p>
          </div>

          {/* Preview thumbnails */}
          <div
            style={{
              display: 'flex',
              gap: 'var(--space-sm)',
              justifyContent: 'center',
              flexWrap: 'wrap',
              width: '100%',
            }}
          >
            {spiralFile && (
              <div style={{ textAlign: 'center' }}>
                <img
                  src={URL.createObjectURL(spiralFile)}
                  alt="Spiral drawing"
                  style={{
                    width: 120,
                    height: 120,
                    objectFit: 'contain',
                    borderRadius: 'var(--radius-card)',
                    border: '1px solid var(--color-rule-2)',
                    background: 'var(--color-paper-3)',
                  }}
                />
                <p
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-ink-3)',
                    fontFamily: 'var(--font-mono)',
                    margin: '4px 0 0',
                  }}
                >
                  spiral
                </p>
              </div>
            )}
            {waveFile && (
              <div style={{ textAlign: 'center' }}>
                <img
                  src={URL.createObjectURL(waveFile)}
                  alt="Wave drawing"
                  style={{
                    width: 120,
                    height: 120,
                    objectFit: 'contain',
                    borderRadius: 'var(--radius-card)',
                    border: '1px solid var(--color-rule-2)',
                    background: 'var(--color-paper-3)',
                  }}
                />
                <p
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-ink-3)',
                    fontFamily: 'var(--font-mono)',
                    margin: '4px 0 0',
                  }}
                >
                  wave
                </p>
              </div>
            )}
          </div>

          {/* Disclaimer */}
          <p
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-3)',
              fontFamily: 'var(--font-mono)',
              textAlign: 'center',
              maxWidth: 500,
              margin: 0,
              lineHeight: 1.7,
            }}
          >
            ⚠ research use only — not a clinical diagnosis. consult a qualified neurologist.
          </p>

          <button
            id="hw-reset-btn"
            onClick={handleReset}
            style={{
              background: 'none',
              border: '1px solid var(--color-rule-2)',
              borderRadius: 'var(--radius-btn)',
              color: 'var(--color-ink-2)',
              cursor: 'pointer',
              padding: '8px 20px',
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              transition: 'all var(--dur-short) var(--ease-out)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-accent)';
              e.currentTarget.style.color = 'var(--color-accent)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-rule-2)';
              e.currentTarget.style.color = 'var(--color-ink-2)';
            }}
          >
            analyse another pair
          </button>
        </div>
      )}

      {/* spin keyframe (inline) */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

export default Handwriting;
