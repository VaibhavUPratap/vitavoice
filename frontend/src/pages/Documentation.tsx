import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Book, FileText, ShieldAlert, HeartHandshake, Box, Map, Info, Terminal, Activity, ChevronRight } from 'lucide-react';

const DOC_FILES = [
  { id: 'architecture.md', icon: Box, label: 'System Architecture', desc: 'Core infrastructure & pipeline' },
  { id: 'model_card.md', icon: Info, label: 'Model Card', desc: 'WavLM & ResNet18 specs' },
  { id: 'dataset_info.md', icon: FileText, label: 'Dataset Information', desc: 'Oxford Parkinson\'s data' },
  { id: 'developer_guide.md', icon: Map, label: 'Developer Guide', desc: 'API & codebase navigation' },
  { id: 'deployment_guide.md', icon: Book, label: 'Deployment Guide', desc: 'Production environment setup' },
  { id: 'ethical_considerations.md', icon: ShieldAlert, label: 'Ethical Considerations', desc: 'Bias & fairness reporting' },
  { id: 'responsible_ai.md', icon: HeartHandshake, label: 'Responsible AI', desc: 'Clinical safety protocols' },
];

export function Documentation({ onBack }: { onBack: () => void }) {
  const [activeDoc, setActiveDoc] = useState<string>(DOC_FILES[0].id);
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [fading, setFading] = useState<boolean>(false);

  useEffect(() => {
    setFading(true);
    const timer = setTimeout(() => {
      setLoading(true);
      fetch(`/docs/${activeDoc}`)
        .then((res) => res.text())
        .then((text) => {
          setContent(text);
          setLoading(false);
          setFading(false);
        })
        .catch(() => {
          setContent('Error loading documentation.');
          setLoading(false);
          setFading(false);
        });
    }, 200);
    return () => clearTimeout(timer);
  }, [activeDoc]);

  return (
    <div
      className="page-wrap reveal is-in doc-container"
      style={{
        width: '100%',
        maxWidth: 1280,
        margin: '0 auto',
        padding: 'var(--space-xl) var(--page-gutter)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-lg)',
        position: 'relative',
      }}
    >
      {/* Background atmospheric mesh */}
      <div 
        style={{
          position: 'absolute',
          top: -100,
          right: -100,
          width: '600px',
          height: '600px',
          background: 'radial-gradient(circle, var(--color-glow) 0%, transparent 60%)',
          opacity: 0.15,
          zIndex: -1,
          pointerEvents: 'none',
        }}
      />

      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', borderBottom: '1px solid var(--color-rule)', paddingBottom: 'var(--space-md)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <button
            onClick={onBack}
            className="doc-back-btn"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-ink-3)',
              cursor: 'pointer',
              padding: 0,
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              transition: 'all var(--dur-short) var(--ease-out)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}
          >
            <ChevronRight style={{ width: 14, height: 14, transform: 'rotate(180deg)' }} />
            Return to Dashboard
          </button>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', marginBottom: '4px' }}>
              <Activity className="vocal-analyzer__icon" style={{ width: 14, height: 14, color: 'var(--color-accent)' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', fontWeight: 600, color: 'var(--color-accent)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Technical Reference
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
              <img src="/logo.svg" alt="VitaVoice" style={{ width: 36, height: 36, borderRadius: 8 }} />
              <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)', color: 'var(--color-ink)', margin: 0, fontWeight: 400, letterSpacing: '-0.02em' }}>
                System Documentation
              </h1>
            </div>
          </div>
        </div>

        <div className="vocal-analyzer__meta-bar" style={{ border: '1px solid var(--color-rule-2)', borderRadius: 'var(--radius-pill)', background: 'var(--color-paper-2)', padding: '6px 16px', display: 'flex', gap: '16px' }}>
          <span>STATUS: ONLINE</span>
          <span>VERSION: 2.1.0</span>
        </div>
      </header>

      <div style={{ display: 'flex', gap: 'var(--space-xl)', alignItems: 'flex-start' }}>
        {/* Sidebar Nav */}
        <nav
          className="doc-sidebar"
          style={{
            width: 280,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            position: 'sticky',
            top: '8rem',
          }}
        >
          {DOC_FILES.map((doc, idx) => {
            const Icon = doc.icon;
            const isActive = activeDoc === doc.id;
            return (
              <button
                key={doc.id}
                onClick={() => setActiveDoc(doc.id)}
                className={`doc-nav-item ${isActive ? 'is-active' : ''}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-md)',
                  padding: '14px 16px',
                  background: isActive ? 'var(--color-paper-2)' : 'transparent',
                  border: '1px solid',
                  borderColor: isActive ? 'var(--color-rule-2)' : 'transparent',
                  borderRadius: 'var(--radius-card)',
                  color: isActive ? 'var(--color-ink)' : 'var(--color-ink-2)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all var(--dur-short) var(--ease-out)',
                  animation: `fade-in-up 0.4s var(--ease-out) ${idx * 0.05}s backwards`,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                {isActive && (
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '3px', background: 'var(--color-accent)', boxShadow: '0 0 10px var(--color-accent)' }} />
                )}
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  width: 32, 
                  height: 32, 
                  borderRadius: '8px', 
                  background: isActive ? 'var(--color-paper-3)' : 'transparent',
                  color: isActive ? 'var(--color-accent)' : 'inherit',
                  transition: 'all var(--dur-short) var(--ease-out)'
                }}>
                  <Icon style={{ width: 16, height: 16 }} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                    {doc.label}
                  </span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--color-ink-3)' }}>
                    {doc.desc}
                  </span>
                </div>
              </button>
            );
          })}
        </nav>

        {/* Content Area */}
        <article
          className="doc-content-area"
          style={{
            flex: 1,
            background: 'var(--color-paper-2)',
            border: '1px solid var(--color-rule)',
            borderRadius: '16px',
            padding: 'clamp(2rem, 5vw, 4rem)',
            minHeight: '70vh',
            boxShadow: 'var(--shadow-subtle)',
            position: 'relative',
            opacity: fading ? 0 : 1,
            transform: fading ? 'translateY(10px)' : 'translateY(0)',
            transition: 'opacity 0.3s var(--ease-out), transform 0.3s var(--ease-out)',
          }}
        >
          {/* Decorative terminal header inside the content */}
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '40px', background: 'var(--color-paper-3)', borderTopLeftRadius: '16px', borderTopRightRadius: '16px', borderBottom: '1px solid var(--color-rule)', display: 'flex', alignItems: 'center', padding: '0 16px', gap: '8px' }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--color-danger)' }} />
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--color-warning)' }} />
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--color-success)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: 'auto', color: 'var(--color-ink-3)', fontFamily: 'var(--font-mono)', fontSize: '0.6rem' }}>
              <Terminal style={{ width: 12, height: 12 }} />
              vitavoice/{activeDoc}
            </div>
          </div>

          <div className="doc-markdown" style={{ marginTop: '30px' }}>
            {loading && !content ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px' }}>
                <span className="vocal-analyzer__led" style={{ width: 8, height: 8, background: 'var(--color-accent)', boxShadow: '0 0 10px var(--color-accent)' }} />
              </div>
            ) : (
              <ReactMarkdown>{content}</ReactMarkdown>
            )}
          </div>
        </article>
      </div>

      <style>{`
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .doc-back-btn:hover {
          color: var(--color-accent) !important;
          transform: translateX(-4px);
        }

        .doc-nav-item:hover:not(.is-active) {
          background: var(--color-paper-emit) !important;
          transform: translateX(4px);
          border-color: var(--color-rule) !important;
        }

        .doc-markdown {
          color: var(--color-ink);
          line-height: 1.8;
          font-size: 0.95rem;
        }
        
        .doc-markdown h1, 
        .doc-markdown h2, 
        .doc-markdown h3 {
          font-family: var(--font-display);
          font-weight: 400;
          color: var(--color-ink);
          margin-top: 2em;
          margin-bottom: 0.5em;
          letter-spacing: -0.01em;
        }

        .doc-markdown h1 { font-size: 2.5rem; margin-top: 0; border-bottom: 1px solid var(--color-rule); padding-bottom: 0.5em; }
        .doc-markdown h2 { font-size: 2rem; color: var(--color-accent); }
        .doc-markdown h3 { font-size: 1.5rem; }

        .doc-markdown p {
          margin-bottom: 1.5em;
          color: var(--color-ink-2);
        }

        .doc-markdown a {
          color: var(--color-accent);
          text-decoration: none;
          border-bottom: 1px solid transparent;
          transition: border-color 0.2s;
        }
        .doc-markdown a:hover {
          border-color: var(--color-accent);
        }

        .doc-markdown ul, .doc-markdown ol {
          margin-bottom: 1.5em;
          padding-left: 1.5em;
          color: var(--color-ink-2);
        }
        
        .doc-markdown li { margin-bottom: 0.5em; }

        .doc-markdown blockquote {
          margin: 2em 0;
          padding: 1em 1.5em;
          border-left: 3px solid var(--color-accent-2);
          background: var(--color-paper-3);
          border-radius: 0 8px 8px 0;
          color: var(--color-ink-3);
          font-style: italic;
        }

        .doc-markdown code {
          font-family: var(--font-mono);
          background: var(--color-paper-3);
          padding: 0.2em 0.4em;
          border-radius: 4px;
          font-size: 0.85em;
          color: var(--color-ink);
          border: 1px solid var(--color-rule);
        }

        .doc-markdown pre {
          background: var(--color-graphite-2);
          padding: 1.5em;
          border-radius: 12px;
          overflow-x: auto;
          border: 1px solid var(--color-rule);
          box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
          margin: 2em 0;
        }
        
        .doc-markdown pre code {
          background: transparent;
          padding: 0;
          border: none;
          color: oklch(90% 0.02 260); /* distinct code color */
        }

        .doc-markdown table {
          width: 100%;
          border-collapse: collapse;
          margin: 2em 0;
        }
        
        .doc-markdown th, .doc-markdown td {
          padding: 12px 16px;
          border-bottom: 1px solid var(--color-rule);
          text-align: left;
        }
        
        .doc-markdown th {
          font-family: var(--font-mono);
          font-size: 0.75rem;
          text-transform: uppercase;
          color: var(--color-ink-3);
          background: var(--color-paper-3);
        }
      `}</style>
    </div>
  );
}

export default Documentation;
