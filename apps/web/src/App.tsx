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
  insurer_display_name?: string | null;
  product_type_display_name?: string | null;
  product_display_name?: string | null;
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

type FilterOption = { value: string; label: string };

type RetrievalOptions = {
  insurers: FilterOption[];
  product_types: FilterOption[];
  product_names: FilterOption[];
  variant_names: FilterOption[];
  policy_unit_names: FilterOption[];
};

type OptionsLoadState = "idle" | "loading" | "ok" | "error";

function normalizeApiBase(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

const PREVIEW_CHARS = 280;

const API_REACH_HELP =
  "검색 API에 연결할 수 없습니다 ({url}).\n\n아래 명령으로 API가 실행 중인지 확인하세요:\n\nuv run uvicorn insurance_ai_api.main:app --host 127.0.0.1 --port 8765";

/** 화면 예시용 정적 질문(검색창에만 채움). 채팅·자동 검색 없음. */
const EXAMPLE_EVIDENCE_QUERIES: readonly string[] = [
  "보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
  "청약 철회는 언제까지 가능해?",
  "보험금을 청구하려면 어떤 서류가 필요해?",
  "해약환급금은 어떻게 지급돼?",
];

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

function isFilterOptionList(value: unknown): value is FilterOption[] {
  if (!Array.isArray(value)) {
    return false;
  }
  return value.every(
    (x) =>
      x !== null &&
      typeof x === "object" &&
      typeof (x as FilterOption).value === "string" &&
      typeof (x as FilterOption).label === "string",
  );
}

function mergeSortedChoiceOptions(current: string, options: FilterOption[]): FilterOption[] {
  const map = new Map(options.map((o) => [o.value, o]));
  const t = current.trim();
  if (t && !map.has(t)) {
    map.set(t, { value: t, label: t });
  }
  return Array.from(map.values()).sort((a, b) => a.value.localeCompare(b.value));
}

function scopeSummaryLine(
  insurer: string,
  productType: string,
  variantName: string,
  options: RetrievalOptions | null,
): string {
  const iv = insurer.trim();
  const pv = productType.trim();
  const vv = variantName.trim();
  const pick = (v: string, list: FilterOption[]) => {
    if (!v) {
      return "전체";
    }
    const hit = list.find((o) => o.value === v);
    return hit?.label ?? v;
  };
  if (!options) {
    return `${iv || "전체"} / ${pv || "전체"} / ${vv || "전체"}`;
  }
  const vLabel =
    (vv && options.variant_names.find((o) => o.value === vv)?.label) || vv || "전체";
  return `${pick(iv, options.insurers)} / ${pick(pv, options.product_types)} / ${vLabel}`;
}

function CitationCard({ c }: { c: CitationEntry }) {
  const [expanded, setExpanded] = useState(false);
  const raw = c.text.trim();
  const isLong = raw.length > PREVIEW_CHARS;
  const preview = isLong && !expanded ? `${raw.slice(0, PREVIEW_CHARS)}…` : raw;

  const insurerLabel = c.insurer_display_name ?? c.insurer;
  const productLabel = c.product_type_display_name ?? c.product_type;
  const prettyMeta = [insurerLabel, productLabel, c.variant_name].filter(Boolean).join(" · ");
  const slugMeta = [c.insurer, c.product_type, c.variant_name].filter(Boolean).join(" · ");
  const showSlugMeta = Boolean(slugMeta && prettyMeta && slugMeta !== prettyMeta);

  return (
    <article className="citation-card">
      <div className="citation-card__id">{c.citation_id}</div>
      <h3 className="citation-card__title">{c.section_title || "(제목 없음)"}</h3>
      <div className="citation-card__row">
        <span className="badge">{c.section_type || "—"}</span>
        <span className="citation-card__pages">
          {c.page_start}–{c.page_end}쪽
        </span>
        <span className="citation-card__score">유사도 {c.score.toFixed(4)}</span>
      </div>
      <div className="citation-card__meta">{prettyMeta || "—"}</div>
      {showSlugMeta ? <div className="citation-card__meta-slug">{slugMeta}</div> : null}
      <p className="citation-card__text">{preview}</p>
      {isLong ? (
        <button type="button" className="link-button" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "접기" : "더 보기"}
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
          !isFilterOptionList(parsed.insurers) ||
          !isFilterOptionList(parsed.product_types) ||
          !isFilterOptionList(parsed.product_names) ||
          !isFilterOptionList(parsed.variant_names) ||
          !isFilterOptionList(parsed.policy_unit_names)
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
    () => mergeSortedChoiceOptions(insurer, retrievalOptions?.insurers ?? []),
    [insurer, retrievalOptions],
  );
  const productTypeChoices = useMemo(
    () => mergeSortedChoiceOptions(productType, retrievalOptions?.product_types ?? []),
    [productType, retrievalOptions],
  );
  const variantChoices = useMemo(
    () => mergeSortedChoiceOptions(variantName, retrievalOptions?.variant_names ?? []),
    [variantName, retrievalOptions],
  );

  const useFilterSelects = optionsLoad === "ok" && retrievalOptions !== null;

  const canSearch = useMemo(() => query.trim().length > 0 && indexDir.trim().length > 0, [
    query,
    indexDir,
  ]);

  const scopeSummary = useMemo(
    () => scopeSummaryLine(insurer, productType, variantName, retrievalOptions),
    [insurer, productType, variantName, retrievalOptions],
  );

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
        setError(`JSON 형식이 아닙니다 (HTTP ${res.status}): ${text.slice(0, 200)}`);
        return;
      }
      if (!res.ok) {
        const detail =
          typeof data === "object" && data !== null && "detail" in data
            ? String((data as { detail: unknown }).detail)
            : text;
        setError(`요청 실패 (HTTP ${res.status}): ${detail}`);
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
        <aside className="sidebar" aria-label="검색 범위 및 인덱스">
          <div className="sidebar__block">
            <h2 className="sidebar__heading">인덱스</h2>
            <label className="field">
              <span className="field__label">인덱스 경로</span>
              <input
                type="text"
                value={indexDir}
                onChange={(e) => setIndexDir(e.target.value)}
                spellCheck={false}
              />
            </label>
            <p className="sidebar__hint">API 서버가 사용할 로컬 검색 인덱스 경로입니다.</p>
          </div>

          <div className="sidebar__block">
            <h2 className="sidebar__heading">검색 범위</h2>
            {optionsLoad === "loading" ? (
              <p className="sidebar__hint sidebar__hint--loading">인덱스에서 필터 목록을 불러오는 중…</p>
            ) : null}
            {optionsLoad === "error" ? (
              <p className="sidebar__warning" role="status">
                인덱스에서 필터 옵션을 불러오지 못했습니다. 값을 직접 입력해 주세요.
              </p>
            ) : null}
            <label className="field">
              <span className="field__label">보험사</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={insurer}
                  onChange={(e) => setInsurer(e.target.value)}
                >
                  <option value="">전체</option>
                  {insurerChoices.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input type="text" value={insurer} onChange={(e) => setInsurer(e.target.value)} />
              )}
            </label>
            <label className="field">
              <span className="field__label">상품 유형</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={productType}
                  onChange={(e) => setProductType(e.target.value)}
                >
                  <option value="">전체</option>
                  {productTypeChoices.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
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
              <span className="field__label">상품/가입 형태</span>
              {useFilterSelects ? (
                <select
                  className="field-select"
                  value={variantName}
                  onChange={(e) => setVariantName(e.target.value)}
                >
                  <option value="">전체</option>
                  {variantChoices.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
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
              <span className="field__label">검색 결과 수</span>
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
              <span className="field__label field__label--inline">같은 조항 중복 제거</span>
            </label>
          </div>
        </aside>

        <main className="main-panel">
          <header className="main-header">
            <h1 className="main-title">보험 약관 근거 검색</h1>
            <p className="main-subtitle">
              약관 원문에서 인용 가능한 근거 구절만 찾아줍니다. 답변을 만들거나 요약하지 않으며, 챗봇이나 생성형
              모델 호출도 없습니다. (API: <code>POST /retrieval/context</code>)
            </p>
          </header>

          <div className="query-bar">
            <div className="example-chips" aria-label="예시 근거 검색">
              <p className="example-chips__title">예시 근거 검색</p>
              <div className="example-chips__list">
                {EXAMPLE_EVIDENCE_QUERIES.map((exampleQuery) => (
                  <button
                    key={exampleQuery}
                    type="button"
                    className="example-chip"
                    onClick={() => setQuery(exampleQuery)}
                  >
                    {exampleQuery}
                  </button>
                ))}
              </div>
            </div>
            <label className="field field--grow query-bar__query">
              <span className="field__label">질문</span>
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
                {loading ? "검색 중…" : "근거 검색"}
              </button>
            </div>
          </div>

          <details className="advanced-panel">
            <summary className="advanced-panel__summary">API 연결 설정</summary>
            <div className="advanced-panel__body">
              <label className="field">
                <span className="field__label">API 기본 URL 덮어쓰기(선택)</span>
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
                비워 두면 빌드 시점의 <code>VITE_API_BASE_URL</code>을 사용합니다(기본값{" "}
                <code>http://127.0.0.1:8765</code>). 예시는 <code>apps/web/.env.example</code>를 참고하세요.
              </p>
            </div>
          </details>

          <div className="main-scroll">
            {error ? (
              <div className="state state--error" role="alert">
                <div className="state__title">요청에 실패했습니다</div>
                <div className="state__body state__body--multiline">{error}</div>
              </div>
            ) : null}

            {loading ? (
              <div className="state state--loading" aria-live="polite">
                <div className="state__title">근거를 가져오는 중…</div>
                <div className="state__body">
                  API 응답을 기다리고 있습니다. 첫 요청에서는 서버에서 임베딩 모델을 불러올 수 있어 시간이 걸릴 수
                  있습니다.
                </div>
              </div>
            ) : null}

            {bundle && !loading ? (
              <>
                <p className="result-summary">
                  근거 {bundle.citations.length}건 · {scopeSummary} · 상위 {bundle.top_k}건
                </p>

                {bundle.citations.length === 0 ? (
                  <div className="state state--empty">
                    <div className="state__title">근거가 없습니다</div>
                    <div className="state__body">
                      이 질문과 검색 범위에서 인덱스가 반환한 근거가 없습니다. 검색 범위를 넓히거나 질문을 바꿔
                      다시 시도해 보세요.
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
                  <summary className="raw-json__summary">원시 JSON (CitationContextBundle)</summary>
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
