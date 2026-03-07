export interface SearchResult {
  repo_id: number;
  repo_path: string;
  source_kind: string;
  source_key: string;
  chunk_index: number;
  locator_id: string | null;
  content: string;
  url: string | null;
  score: number | null;
}

export async function searchContent(
  content: string,
  page: number,
  pageSize: number,
): Promise<SearchResult[]> {
  const response = await fetch("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      content,
      page,
      page_size: pageSize,
      top_k: pageSize,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `search failed with status ${response.status}`);
  }

  return (await response.json()) as SearchResult[];
}
