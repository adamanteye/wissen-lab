import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { searchContent, type SearchResult } from "./utils";
import "./styles.css";

const PAGE_SIZE = 30;
const CONTENT_CHAR_LIMIT = 600;
const CONTENT_LINE_LIMIT = 12;
const COMMIT_SHA_DISPLAY_LENGTH = 8;

function formatScore(score: number | null) {
  if (score === null) {
    return "-";
  }

  return score.toFixed(3);
}

function resultKey(result: SearchResult) {
  return [
    result.project_id,
    result.source_kind,
    result.source_key,
    result.chunk_index,
  ].join(":");
}

function formatSourceKey(result: SearchResult) {
  if (result.source_kind !== "commit") {
    return result.source_key;
  }

  return result.source_key.slice(0, COMMIT_SHA_DISPLAY_LENGTH);
}

function isLargeContent(content: string) {
  return (
    content.length > CONTENT_CHAR_LIMIT ||
    content.split(/\r?\n/).length > CONTENT_LINE_LIMIT
  );
}

function resizeTextarea(element: HTMLTextAreaElement) {
  element.style.height = "auto";
  element.style.height = `${element.scrollHeight}px`;
}

function SearchPage() {
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const hasNextPage = results.length === PAGE_SIZE;

  useEffect(() => {
    if (textareaRef.current) {
      resizeTextarea(textareaRef.current);
    }
  }, [draft]);

  async function runSearch(nextQuery: string, nextPage: number) {
    setLoading(true);
    setError("");
    setQuery(nextQuery);
    setPage(nextPage);
    setResults([]);
    setExpandedRows({});

    try {
      const nextResults = await searchContent(nextQuery, nextPage, PAGE_SIZE);
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
      setError("enter search text");
      setResults([]);
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

  function toggleExpanded(rowKey: string) {
    setExpandedRows((current) => ({
      ...current,
      [rowKey]: !current[rowKey],
    }));
  }

  return (
    <main className="wissen">
      <h1 className="title">WissenLab GitLab embedding search</h1>

      <form className="query-line" onSubmit={onSubmit}>
        <textarea
          id="search-query"
          ref={textareaRef}
          className="query-input query-textarea"
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            resizeTextarea(event.target);
          }}
          rows={1}
          spellCheck={false}
          aria-label="Search query"
        />
        <button type="submit" disabled={loading}>
          {loading ? "searching" : "search"}
        </button>
      </form>

      <div className="results-scroll">
        <table className="result-table">
          <colgroup>
            <col className="col-project" />
            <col className="col-source" />
            <col className="col-content" />
            <col className="col-score" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col">project</th>
              <th scope="col">source</th>
              <th scope="col">content</th>
              <th scope="col">score</th>
            </tr>
          </thead>
          <tbody>
            {error ? (
              <tr className="message-row">
                <td className="error-cell" colSpan={4}>
                  {error}
                </td>
              </tr>
            ) : results.length > 0 ? (
              results.map((result) => {
                const rowKey = resultKey(result);
                const contentIsLarge = isLargeContent(result.content);
                const isExpanded = Boolean(expandedRows[rowKey]);

                return (
                  <tr key={rowKey}>
                    <td className="project-cell">
                      <strong>{result.project_path}</strong>
                    </td>
                    <td className="source-cell">
                      <strong>{result.source_kind}</strong>
                      <span title={result.source_key}>
                        #{formatSourceKey(result)}
                      </span>
                      {result.locator_id ? (
                        <span>locator {result.locator_id}</span>
                      ) : null}
                    </td>
                    <td className="content-cell">
                      <pre
                        className={
                          contentIsLarge && !isExpanded ? "content-preview" : ""
                        }
                      >
                        {result.content}
                      </pre>
                      {contentIsLarge ? (
                        <button
                          type="button"
                          className="content-toggle"
                          onClick={() => toggleExpanded(rowKey)}
                        >
                          {isExpanded ? "hide" : "expand"}
                        </button>
                      ) : null}
                    </td>
                    <td className="score-cell">
                      {result.url ? (
                        <a href={result.url} target="_blank" rel="noreferrer">
                          {formatScore(result.score)}
                        </a>
                      ) : (
                        formatScore(result.score)
                      )}
                    </td>
                  </tr>
                );
              })
            ) : query && !loading ? (
              <tr className="message-row">
                <td colSpan={4}>no results.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="pager">
        <button
          type="button"
          onClick={() => openPage(page - 1)}
          disabled={loading || page <= 1}
        >
          prev
        </button>
        <span>page {page}</span>
        <button
          type="button"
          onClick={() => openPage(page + 1)}
          disabled={loading || !hasNextPage}
        >
          next
        </button>
      </div>
    </main>
  );
}

export default SearchPage;
