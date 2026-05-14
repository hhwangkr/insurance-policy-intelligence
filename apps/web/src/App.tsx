import { useCallback, useEffect, useMemo, useState } from "react";

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

type RetrievalOptions = {
  insurers: string[];
  product_types: string[];
  product_names: string[];
  variant_names: string[];
  policy_unit_names: string[];
};

type OptionsLoadState = "idle" | "loading" | "ok" | "error";

function normalizeApiBase(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

const PREVIEW_CHARS = 280;

const API_REACH_HELP =
  "Could not reach the retrieval API at {url}. Make sure the API is running:\n\nuv run uvicorn insurance_ai_api.main:app --host 127.0.0.1 --port 8765";

function isLikelyFetchNetworkError(e: unknown): boolean {
  if (e instanceof TypeError) {
    return true;
  }
  if (e instanceof Error) {
    if (e.name === "AbortError") {
      return true;
    }
    return /failed to fetch|networkerror|load failed|network request failed/i.test(e.message);
  }
  return false;
}

function mergeSortedChoices(current: string, apiList: string[]): string[] {
  const set = new Set(apiList);
  const t = current.trim();
  if (t) {
    set.add(t);
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}

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
  const viteApiDefault = useMemo(
    () => normalizeApiBase(import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8765"),
    [],
  );
  const [apiBaseOverride, setApiBaseOverride] = useState("");
  const apiBaseUrl = useMemo(
    () =>
      apiBaseOverride.trim() ? normalizeApiBase(apiBaseOverride) : viteApiDefault,
    [apiBaseOverride, viteApiDefault],
  );

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

  const [retrievalOptions, setRetrievalOptions] = useState<RetrievalOptions | null>(null);
  const [optionsLoad, setOptionsLoad] = useState<OptionsLoadState>("idle");

  useEffect(() => {
    if (!indexDir.trim()) {
      setOptionsLoad("idle");
      setRetrievalOptions(null);
      return;
    }
    let cancelled = false;
    setOptionsLoad("loading");
    const params = new URLSearchParams({ index_dir: indexDir.trim() });
    const url = `${apiBaseUrl}/retrieval/options?${params.toString()}`;
    void fetch(url)
      .then(async (res) => {
        if (cancelled) {
          return;
        }
        if (!res.ok) {
          setRetrievalOptions(null);
          setOptionsLoad("error");
          return;
        }
        let data: unknown;
        try {
          data = await res.json();
        } catch {
          setRetrievalOptions(null);
          setOptionsLoad("error");
          return;
        }
        if (cancelled) {
          return;
        }
        const parsed = data as Partial<RetrievalOptions>;
        if (
          !Array.isArray(parsed.insurers) ||
          !Array.isArray(parsed.product_types) ||
          !Array.isArray(parsed.product_names) ||
          !Array.isArray(parsed.variant_names) ||
          !Array.isArray(parsed.policy_unit_names)
        ) {
          setRetrievalOptions(null);
          setOptionsLoad("error");
          return;
        }
        setRetrievalOptions({
          insurers: parsed.insurers,
          product_types: parsed.product_types,
          product_names: parsed.product_names,
          variant_names: parsed.variant_names,
          policy_unit_names: parsed.policy_unit_names,
        });
        setOptionsLoad("ok");
      })
      .catch(() => {
        if (!cancelled) {
          setRetrievalOptions(null);
          setOptionsLoad("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, indexDir]);

  const insurerChoices = useMemo(
    () => mergeSortedChoices(insurer, retrievalOptions?.insurers ?? []),
    [insurer, retrievalOptions],
  );
  const productTypeChoices = useMemo(
    () => mergeSortedChoices(productType, retrievalOptions?.product_types ?? []),
    [productType, retrievalOptions],
  );
  const variantChoices = useMemo(
    () => mergeSortedChoices(variantName, retrievalOptions?.variant_names ?? []),
    [variantName, retrievalOptions],
  );

  const useFilterSelects = optionsLoad === "ok" && retrievalOptions !== null;

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
    const url = `${apiBaseUrl}/retrieval/context`;
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
      if (isLikelyFetchNetworkError(e)) {
        setError(API_REACH_HELP.replace("{url}", apiBaseUrl));
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setLoading(false);
    }
  }, [
    apiBaseUrl,
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
            <h2 className="sidebar__heading">Index</h2>
            <label className="field">
              <span className="field__label">index_dir (server)</span>
              <input
                type="text"
                value={indexDir}
                onChange={(e) => setIndexDir(e.target.value)}
                spellCheck={false}
              />
            </label>
            <p className="sidebar__hint">Path as seen by the API process (repo root when running uvicorn locally).</p>
          </div>

          <div className="sidebar__block">
            <h2 className="sidebar__heading">Filters</h2>
            {optionsLoad === "loading" ? (
              <p className="sidebar__hint sidebar__hint--loading">Loading filter values from index…</p>
            ) : null}
            {optionsLoad === "error" ? (
              <p className="sidebar__warning" role="status">
                Could not load filter options from the index. Enter values manually.
              </p>
            ) : null}
            <label className="field">
              <span className="field__label">insurer</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={insurer}
                  onChange={(e) => setInsurer(e.target.value)}
                >
                  <option value="">Any</option>
                  {insurerChoices.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              ) : (
                <input type="text" value={insurer} onChange={(e) => setInsurer(e.target.value)} />
              )}
            </label>
            <label className="field">
              <span className="field__label">product_type</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={productType}
                  onChange={(e) => setProductType(e.target.value)}
                >
                  <option value="">Any</option>
                  {productTypeChoices.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={productType}
                  onChange={(e) => setProductType(e.target.value)}
                />
              )}
            </label>
            <label className="field">
              <span className="field__label">variant_name</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={variantName}
                  onChange={(e) => setVariantName(e.target.value)}
                >
                  <option value="">Any</option>
                  {variantChoices.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={variantName}
                  onChange={(e) => setVariantName(e.target.value)}
                />
              )}
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

          <details className="advanced-panel">
            <summary className="advanced-panel__summary">Advanced</summary>
            <div className="advanced-panel__body">
              <label className="field">
                <span className="field__label">API base URL override (optional)</span>
                <input
                  type="text"
                  value={apiBaseOverride}
                  onChange={(e) => setApiBaseOverride(e.target.value)}
                  placeholder={viteApiDefault}
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
              <p className="advanced-panel__hint">
                Leave empty to use <code>VITE_API_BASE_URL</code> from the build (default{" "}
                <code>http://127.0.0.1:8765</code>). See <code>apps/web/.env.example</code>.
              </p>
            </div>
          </details>

          <div className="main-scroll">
            {error ? (
              <div className="state state--error" role="alert">
                <div className="state__title">Request failed</div>
                <div className="state__body state__body--multiline">{error}</div>
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
