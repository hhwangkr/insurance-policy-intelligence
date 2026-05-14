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

function snip(text: string, max = 480): string {
  const t = text.trim();
  if (t.length <= max) {
    return t;
  }
  return `${t.slice(0, max)}…`;
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
    <div className="app">
      <h1>Insurance Policy Evidence Search</h1>
      <p className="subtitle">
        Retrieves citation-ready passages from your local index via{" "}
        <code>POST /retrieval/context</code>. This is evidence search, not a chatbot and not
        LLM-generated answers.
      </p>

      <section className="panel">
        <h2>API &amp; index</h2>
        <div className="form-grid">
          <label>
            API base URL
            <input
              type="text"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <label>
            index_dir (server path)
            <input
              type="text"
              value={indexDir}
              onChange={(e) => setIndexDir(e.target.value)}
              spellCheck={false}
            />
          </label>
        </div>
        <p className="subtitle" style={{ marginTop: "0.5rem", marginBottom: 0 }}>
          Override default URL with <code>VITE_API_BASE_URL</code> at build time if needed.
        </p>
      </section>

      <section className="panel">
        <h2>Search</h2>
        <div className="form-grid">
          <label style={{ gridColumn: "1 / -1" }}>
            Query
            <textarea value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
          <label>
            insurer
            <input type="text" value={insurer} onChange={(e) => setInsurer(e.target.value)} />
          </label>
          <label>
            product_type
            <input
              type="text"
              value={productType}
              onChange={(e) => setProductType(e.target.value)}
            />
          </label>
          <label>
            variant_name
            <input
              type="text"
              value={variantName}
              onChange={(e) => setVariantName(e.target.value)}
            />
          </label>
          <label>
            top_k
            <input
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number.parseInt(e.target.value, 10) || 1)}
            />
          </label>
          <label className="checkbox-row" style={{ alignSelf: "end" }}>
            <input
              type="checkbox"
              checked={dedupeSection}
              onChange={(e) => setDedupeSection(e.target.checked)}
            />
            dedupe_section
          </label>
        </div>
        <div className="actions">
          <button type="button" className="primary" disabled={!canSearch || loading} onClick={runSearch}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
      </section>

      {error ? (
        <div className="banner error" role="alert">
          {error}
        </div>
      ) : null}

      {loading ? (
        <div className="banner info" aria-live="polite">
          Loading… (first request may load the embedding model on the API server.)
        </div>
      ) : null}

      {bundle && !loading ? (
        <>
          {bundle.citations.length === 0 ? (
            <div className="banner info">No citations returned (empty result).</div>
          ) : (
            <section className="panel">
              <h2>Evidence ({bundle.citations.length})</h2>
              <div className="cards">
                {bundle.citations.map((c) => (
                  <article key={c.citation_id} className="card">
                    <header>
                      <span className="cite-id">{c.citation_id}</span>
                      <span className="meta">{c.section_title}</span>
                    </header>
                    <div className="meta">
                      {c.section_type} · p.{c.page_start}–{c.page_end} · score{" "}
                      {c.score.toFixed(4)}
                    </div>
                    <div className="meta">
                      {[c.insurer, c.product_type, c.variant_name].filter(Boolean).join(" · ") ||
                        "—"}
                    </div>
                    <p className="preview">{snip(c.text)}</p>
                  </article>
                ))}
              </div>
            </section>
          )}

          <details className="raw panel">
            <summary>Raw CitationContextBundle JSON</summary>
            <pre>{JSON.stringify(bundle, null, 2)}</pre>
          </details>
        </>
      ) : null}
    </div>
  );
}
