import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Dataset } from "@shared/schema";

interface SearchResult {
  dataset: Dataset;
  score: number;
}

interface SearchResponse {
  query: string;
  processedQuery: string;
  results: SearchResult[];
  timing: {
    total: number;
    processing: number;
    search: number;
  };
}

export const useSearch = () => {
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const queryClient = useQueryClient();

  const search = async (query: string) => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    try {
      const response = await apiRequest("POST", "/api/search", { query });
      const data = await response.json();
      setSearchResults(data);
      
      // Prefetch metadata for top results
      const topResults = data.results.slice(0, 3);
      topResults.forEach(result => {
        queryClient.prefetchQuery({
          queryKey: ['/api/datasets', result.dataset.id, 'metadata']
        });
      });
      
      return data;
    } catch (error) {
      console.error("Search error:", error);
      throw error;
    } finally {
      setIsSearching(false);
    }
  };

  const clearSearch = () => {
    setSearchResults(null);
  };

  return {
    search,
    clearSearch,
    searchResults,
    isSearching
  };
};
