import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { searchContent, type SearchResult } from "./utils";
import "./styles.css";

const PAGE_SIZE = 10;

function formatScore(score: number | null) {
  if (score === null) {
    return "";
  }

  return score.toFixed(3);
}

function SearchPage() {
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hasNextPage = results.length === PAGE_SIZE;
  const summary = useMemo(() => {
    if (!query) {
      return "Search GitLab facts.";
    }

    if (loading) {
      return `Searching page ${page}...`;
    }

    return `${results.length} results on page ${page}.`;
  }, [loading, page, query, results.length]);

  async function runSearch(nextQuery: string, nextPage: number) {
    setLoading(true);
    setError("");

    try {
      const nextResults = await searchContent(nextQuery, nextPage, PAGE_SIZE);
      setQuery(nextQuery);
      setPage(nextPage);
      setResults(nextResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : "search failed");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextQuery = draft.trim();
    if (!nextQuery) {
      setError("Enter search text.");
      return;
    }

    void runSearch(nextQuery, 1);
  }

  function openPage(nextPage: number) {
    if (!query || nextPage < 1 || loading) {
      return;
    }

    void runSearch(query, nextPage);
  }

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">Wissen Lab</p>
        <h1>Search GitLab facts</h1>
        <p className="lede">
          Issues, merge requests, commits. One box. Source links included.
        </p>
      </section>

      <section className="panel">
        <form className="searchbar" onSubmit={onSubmit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            rows={4}
          />
          <div className="actions">
            <button type="submit" disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </form>

        <div className="statusline">
          <span>{summary}</span>
          {error ? <span className="error">{error}</span> : null}
        </div>

        <div className="results">
          {results.map((result) => (
            <article className="result" key={resultKey(result)}>
              <div className="resulthead">
                <div>
                  <strong>{result.repo_path}</strong>
                  <span className="kind">
                    {result.source_kind} #{result.source_key}
                  </span>
                </div>
                <div className="meta">
                  {result.score !== null ? (
                    <span>score {formatScore(result.score)}</span>
                  ) : null}
                  {result.url ? (
                    <a href={result.url} target="_blank" rel="noreferrer">
                      open
                    </a>
                  ) : null}
                </div>
              </div>
              <p>{result.content}</p>
            </article>
          ))}

          {!loading && query && results.length === 0 ? (
            <div className="empty">No results.</div>
          ) : null}
        </div>

        <div className="pager">
          <button
            type="button"
            onClick={() => openPage(page - 1)}
            disabled={loading || page <= 1}
          >
            Previous
          </button>
          <span>Page {page}</span>
          <button
            type="button"
            onClick={() => openPage(page + 1)}
            disabled={loading || !hasNextPage}
          >
            Next
          </button>
        </div>
      </section>
    </main>
  );
}

function resultKey(result: SearchResult) {
  return [
    result.repo_id,
    result.source_kind,
    result.source_key,
    result.chunk_index,
  ].join(":");
}

export default SearchPage;
