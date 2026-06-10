# Design — VitaVoice

A locked design system for this app. Every page redesign reads this file before
emitting code. Do not regenerate per page — extend or amend this file when the
system needs to grow.

## Genre
modern-minimal

## Macrostructure family

- Marketing / landing: Stat-Led — hero pairs a real pipeline metric with a worded headline; supporting stats qualify the hybrid feature stack.
- App / workflow screens: Workbench — functional headings, bordered panels, step sequences; function carries the page.
- Results dashboard: Workbench — tabular spec sheets, hairline grids, no glass or glow.

## Theme
- `--color-paper`   oklch(98.5% 0.004 250)
- `--color-paper-2` oklch(96% 0.006 252)
- `--color-ink`     oklch(24% 0.02 258)
- `--color-ink-2`   oklch(34% 0.018 257)
- `--color-rule`    oklch(90% 0.008 255)
- `--color-accent`  oklch(58% 0.20 256)
- `--color-focus`   oklch(58% 0.20 256)

## Typography
- Display: Space Grotesk, weight 600, style normal
- Body:    Inter, weight 400
- Mono:    JetBrains Mono, weight 500
- Display tracking: -0.02em
- Type scale anchor: `--text-display` = clamp(2.25rem, 4vw + 0.5rem, 3.75rem)

## Spacing
4-point named scale. The values are in `frontend/tokens.css`. Pages must use named
tokens (`var(--space-md)`), never raw values.

## Motion
- Easings: cubic-bezier(0.16, 1, 0.3, 1) named `--ease-out`, etc.
- Reveal pattern: fade + 10px rise on section entry; counter tick on hero stat only.
- Reduced-motion fallback: opacity-only, ≤ 150 ms.

## Microinteractions stance
- Silent success — no celebratory toasts
- Hover delay 800 ms on tooltips · focus delay 0 ms
- Border-colour shift on interactive surfaces; no bounce or parallax

## CTA voice
- Primary CTA: solid cobalt fill, 6px radius, Space Grotesk 600, action verbs ("Start Voice Assessment")
- Secondary CTA: hairline outline, typographic link with arrow

## Per-page allowances
- Marketing pages MAY use a single graphite pipeline card (Tier-A, code-as-hero).
- App pages MUST NOT use enrichment — function carries the page.
- Results dashboard: typography + charts only.

## What pages MUST share
- The VitaVoice wordmark (Space Grotesk).
- Cobalt accent at ≤ 5% per viewport.
- Space Grotesk + Inter + JetBrains Mono pairing.
- 6px button radius, hairline borders, no glassmorphism.
- Mono uppercase labels for meta and status.

## What pages MAY differ on
- Macrostructure within family (Stat-Led landing vs Workbench recorder/results).
- Section layout density (landing is airy; dashboard is dense).

## Exports

See [`export-formats.md`](.agents/skills/hallmark/references/export-formats.md) for canonical mapping.

### tokens.css
```css
:root {
  --color-paper:      oklch(98.5% 0.004 250);
  --color-paper-2:    oklch(96% 0.006 252);
  --color-paper-3:    oklch(93% 0.008 254);
  --color-graphite:   oklch(22% 0.016 260);
  --color-ink:        oklch(24% 0.02 258);
  --color-ink-2:      oklch(34% 0.018 257);
  --color-ink-3:      oklch(52% 0.012 256);
  --color-rule:       oklch(90% 0.008 255);
  --color-rule-2:     oklch(82% 0.010 256);
  --color-accent:     oklch(58% 0.20 256);
  --color-accent-ink: oklch(99% 0.004 250);
  --color-focus:      oklch(58% 0.20 256);
  --color-success:    oklch(52% 0.14 155);
  --color-warning:    oklch(62% 0.14 75);
  --color-danger:     oklch(52% 0.18 25);

  --font-display: "Space Grotesk", system-ui, sans-serif;
  --font-body:    "Inter", system-ui, sans-serif;
  --font-mono:    "JetBrains Mono", ui-monospace, monospace;

  --space-3xs: 0.25rem;  --space-2xs: 0.5rem;  --space-xs: 0.75rem;
  --space-sm:  1rem;     --space-md:  1.5rem;  --space-lg: 2rem;
  --space-xl:  3rem;     --space-2xl: 4.5rem;  --space-3xl: 7rem;

  --text-xs: 0.75rem;  --text-sm: 0.875rem; --text-md: 1.125rem;
  --text-lg: 1.375rem; --text-xl: 1.75rem;  --text-2xl: 2.25rem;

  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --dur-short: 220ms;
  --radius-card: 10px; --radius-btn: 6px; --radius-input: 6px;
}
```

### Tailwind v4 `@theme`
```css
@theme {
  --color-paper:   oklch(98.5% 0.004 250);
  --color-ink:     oklch(24% 0.02 258);
  --color-accent:  oklch(58% 0.20 256);
  --font-display:  "Space Grotesk", sans-serif;
  --font-body:     "Inter", sans-serif;
  --spacing-md:    1.5rem;
  --text-md:       1.125rem;
  --ease-out:      cubic-bezier(0.16, 1, 0.3, 1);
}
```

### DTCG `tokens.json`
```json
{
  "color": {
    "paper":  { "$value": "oklch(98.5% 0.004 250)", "$type": "color" },
    "ink":    { "$value": "oklch(24% 0.02 258)", "$type": "color" },
    "accent": { "$value": "oklch(58% 0.20 256)", "$type": "color" }
  },
  "font": {
    "display": { "$value": "Space Grotesk", "$type": "fontFamily" },
    "body":    { "$value": "Inter", "$type": "fontFamily" }
  },
  "space": {
    "md": { "$value": "1.5rem", "$type": "dimension" }
  }
}
```

### shadcn/ui CSS variables
```css
:root {
  --background:         98.5% 0.004 250;
  --foreground:         24% 0.02 258;
  --primary:            58% 0.20 256;
  --primary-foreground: 99% 0.004 250;
  --muted:              90% 0.008 255;
  --muted-foreground:   52% 0.012 256;
  --border:             90% 0.008 255;
  --input:              90% 0.008 255;
  --ring:               58% 0.20 256;
  --radius:             6px;
}
```
