# Design — VitaVoice

Locked design system. Future Hallmark runs read this file first; pages defer
to it. Amend intentionally — the file is the rule.

## System
- Genre · atmospheric
- Macrostructure · Marquee Hero + Workbench
- Theme · lumen (Night Foundry drop)
- Axes · Night (dark cool-violet) / display-style: classical-serif-lowercase / accent-hue: molten brass

## Tokens (canonical · `tokens.css` is the source of truth)
```css
:root {
  --color-paper:      oklch(13% 0.014 265);
  --color-paper-2:    oklch(17% 0.016 265);
  --color-paper-3:    oklch(22% 0.018 265);
  --color-ink:        oklch(96% 0.006 262);
  --color-ink-2:      oklch(84% 0.008 262);
  --color-ink-3:      oklch(62% 0.010 262);
  --color-rule:       oklch(22% 0.014 265);
  --color-rule-2:     oklch(28% 0.016 265);
  --color-accent:     oklch(76% 0.17 50);
  --color-accent-2:   oklch(68% 0.16 18);
  --color-glow:       oklch(80% 0.16 50 / 0.42);
  --color-paper-emit: oklch(76% 0.17 50 / 0.04);
  --rule-blueprint:   oklch(96% 0.006 262 / 0.04);

  --font-display: "Instrument Serif", ui-serif, Georgia, serif;
  --font-body:    "Inter", system-ui, sans-serif;
  --font-mono:    "JetBrains Mono", ui-monospace, monospace;

  /* 4-pt spacing scale, named: --space-3xs … --space-4xl. See tokens.css.   */
  /* Type scale, 1.25 (major-third) ratio: --text-xs … --text-display.       */

  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --dur-fast: 180ms;  --dur-base: 240ms;  --dur-slow: 320ms;

  --radius-card: 10px;  --radius-pill: 9999px;  --radius-input: 6px;
}
```

## CTA voice
- Primary · oklch(76% 0.17 50) molten brass · 6px · lowercase prose
- Secondary · outline oklch(28% 0.016 265) · 6px

## Motion stance
- filament pulse, 320ms reveal on verb-landmark pivot underline, static meter.
- Reduced-motion fallback · ≤150 ms opacity crossfade.

## Exports
`tokens.css` (in this project) is the source of truth.
