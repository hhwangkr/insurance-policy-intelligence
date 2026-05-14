import { useCallback, useMemo, useState } from "react";

import "./App.css";

type CitationEntry = {
  citation_id: string;
  chunk_id: string;
  section_id: string;
  section_title: string;
  section_type: string;
  document_id: string;
  insurer: string | null;
  product_type: string | null;
  policy_unit_name: string | null;
  variant_name: string | null;
  page_start: number;
  page_end: number;
  char_start: number;
  char_end: number;
  score: number;
  text: string;
};

type CitationBundle = {
  query: string;
  filters: Record<string, unknown>;
  top_k: number;
  dedupe_section: boolean;
  citations: CitationEntry[];
};

const defaultApiBase = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8765";

const PREVIEW_CHARS = 280;

function CitationCard({ c }: { c: CitationEntry }) {
  const [expanded, setExpanded] = useState(false);
  const raw = c.text.trim();
  const isLong = raw.length > PREVIEW_CHARS;
  const preview = isLong && !expanded ? `${raw.slice(0, PREVIEW_CHARS)}…` : raw;

  return (
    <article className="citation-card">
      <div className="citation-card__id">{c.citation_id}</div>
      <h3 className="citation-card__title">{c.section_title || "—"}</h3>
      <div className="citation-card__row">
        <span className="badge">{c.section_type || "—"}</span>
        <span className="citation-card__pages">
          p.{c.page_start}–{c.page_end}
        </span>
        <span className="citation-card__score">score {c.score.toFixed(4)}</span>
      </div>
      <div className="citation-card__meta">
        {[c.insurer, c.product_type, c.variant_name].filter(Boolean).join(" · ") || "—"}
      </div>
      <p className="citation-card__text">{preview}</p>
      {isLong ? (
        <button type="button" className="link-button" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </article>
  );
}

export default function App() {
  const [apiBase, setApiBase] = useState(defaultApiBase);
  const [indexDir, setIndexDir] = useState("data/processed/index");
  const [query, setQuery] = useState(
    "보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
  );
  const [insurer, setInsurer] = useState("kyobolife");
  const [productType, setProductType] = useState("annuity");
  const [variantName, setVariantName] = useState("적립형");
  const [topK, setTopK] = useState(5);
  const [dedupeSection, setDedupeSection] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<CitationBundle | null>(null);

  const canSearch = useMemo(() => query.trim().length > 0 && indexDir.trim().length > 0, [
    query,
    indexDir,
  ]);

  const scopeSummary = useMemo(() => {
    const i = insurer.trim() || "—";
    const p = productType.trim() || "—";
    const v = variantName.trim() || "—";
    return `${i} / ${p} / ${v}`;
  }, [insurer, productType, variantName]);

  const runSearch = useCallback(async () => {
    setError(null);
    setBundle(null);
    setLoading(true);
    const base = apiBase.replace(/\/+$/, "");
    const url = `${base}/retrieval/context`;
    const body = {
      query: query.trim(),
      index_dir: indexDir.trim(),
      filters: {
        document_id: null,
        insurer: insurer.trim() || null,
        product_type: productType.trim() || null,
        product_name: null,
        policy_unit_name: null,
        variant_name: variantName.trim() || null,
        include_section_types: null,
        exclude_section_types: [] as string[],
        use_default_section_type_excludes: true,
      },
      top_k: topK,
      dedupe_section: dedupeSection,
    };
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      let data: unknown = null;
      try {
        data = JSON.parse(text) as unknown;
      } catch {
        setError(`Invalid JSON (${res.status}): ${text.slice(0, 200)}`);
        return;
      }
      if (!res.ok) {
        const detail =
          typeof data === "object" && data !== null && "detail" in data
            ? String((data as { detail: unknown }).detail)
            : text;
        setError(`HTTP ${res.status}: ${detail}`);
        return;
      }
      setBundle(data as CitationBundle);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [
    apiBase,
    dedupeSection,
    indexDir,
    insurer,
    productType,
    query,
    topK,
    variantName,
  ]);

  return (
    <div className="app-shell">
      <div className="app-inner">
        <aside className="sidebar" aria-label="Search scope and filters">
          <div className="sidebar__block">
            <h2 className="sidebar__heading">Connection</h2>
            <label className="field">
              <span className="field__label">API base URL</span>
              <input
                type="text"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label className="field">
              <span className="field__label">index_dir (server)</span>
              <input
                type="text"
                value={indexDir}
                onChange={(e) => setIndexDir(e.target.value)}
                spellCheck={false}
              />
            </label>
            <p className="sidebar__hint">
              Optional build override: <code>VITE_API_BASE_URL</code>
            </p>
          </div>

          <div className="sidebar__block">
            <h2 className="sidebar__heading">Filters</h2>
            <label className="field">
              <span className="field__label">insurer</span>
              <input type="text" value={insurer} onChange={(e) => setInsurer(e.target.value)} />
            </label>
            <label className="field">
              <span className="field__label">product_type</span>
              <input
                type="text"
                value={productType}
                onChange={(e) => setProductType(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field__label">variant_name</span>
              <input
                type="text"
                value={variantName}
                onChange={(e) => setVariantName(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field__label">top_k</span>
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number.parseInt(e.target.value, 10) || 1)}
              />
            </label>
            <label className="field field--row">
              <input
                type="checkbox"
                checked={dedupeSection}
                onChange={(e) => setDedupeSection(e.target.checked)}
              />
              <span className="field__label field__label--inline">dedupe_section</span>
            </label>
          </div>
        </aside>

        <main className="main-panel">
          <header className="main-header">
            <h1 className="main-title">Insurance Policy Evidence Search</h1>
            <p className="main-subtitle">
              Evidence console: <code>POST /retrieval/context</code> returns citation-ready passages
              from your local index. No chatbot and no LLM-authored answers.
            </p>
          </header>

          <div className="query-bar">
            <label className="field field--grow">
              <span className="field__label">Query</span>
              <textarea
                className="query-textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
              />
            </label>
            <div className="query-actions">
              <button
                type="button"
                className="btn-primary"
                disabled={!canSearch || loading}
                onClick={runSearch}
              >
                {loading ? "Searching…" : "Search"}
              </button>
            </div>
          </div>

          <div className="main-scroll">
            {error ? (
              <div className="state state--error" role="alert">
                <div className="state__title">Request failed</div>
                <div className="state__body">{error}</div>
              </div>
            ) : null}

            {loading ? (
              <div className="state state--loading" aria-live="polite">
                <div className="state__title">Retrieving evidence…</div>
                <div className="state__body">
                  Waiting for the API. The first call may load the embedding model on the server.
                </div>
              </div>
            ) : null}

            {bundle && !loading ? (
              <>
                <p className="result-summary">
                  {bundle.citations.length} citation{bundle.citations.length === 1 ? "" : "s"}{" "}
                  found · {scopeSummary} · top_k={bundle.top_k}
                </p>

                {bundle.citations.length === 0 ? (
                  <div className="state state--empty">
                    <div className="state__title">No citations</div>
                    <div className="state__body">
                      The index returned an empty list for this query and filter scope. Try
                      widening filters or changing the query.
                    </div>
                  </div>
                ) : (
                  <div className="citation-list">
                    {bundle.citations.map((c) => (
                      <CitationCard key={c.citation_id} c={c} />
                    ))}
                  </div>
                )}

                <details className="raw-json">
                  <summary className="raw-json__summary">Raw CitationContextBundle JSON</summary>
                  <pre className="raw-json__pre">{JSON.stringify(bundle, null, 2)}</pre>
                </details>
              </>
            ) : null}
          </div>
        </main>
      </div>
    </div>
  );
}
