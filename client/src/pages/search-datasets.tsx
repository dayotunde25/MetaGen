import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import DatasetCard from "@/components/dataset-card";
import SearchForm from "@/components/search-form";
import { Dataset } from "@shared/schema";
import { Loader2 } from "lucide-react";

export default function SearchDatasets() {
  const [location] = useLocation();
  const [searchParams, setSearchParams] = useState<URLSearchParams>(new URLSearchParams());
  const [query, setQuery] = useState<string>("");
  const [tags, setTags] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<string>("dateAdded");
  const [page, setPage] = useState<number>(1);
  const pageSize = 9;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setSearchParams(params);
    setQuery(params.get("q") || "");
    setTags(params.get("tags") ? params.get("tags")!.split(",") : []);
    setSortBy(params.get("sortBy") || "dateAdded");
    setPage(parseInt(params.get("page") || "1"));
  }, [location]);

  // Fetch datasets with search parameters
  const { data: datasets, isLoading, refetch } = useQuery<Dataset[]>({
    queryKey: [`/api/search?q=${query}&tags=${tags.join(',')}&sortBy=${sortBy}&limit=${pageSize}&offset=${(page - 1) * pageSize}`],
    enabled: searchParams.toString() !== "",
  });

  const handleSearch = (newQuery: string, newTags: string[]) => {
    const params = new URLSearchParams();
    if (newQuery) params.set("q", newQuery);
    if (newTags.length > 0) params.set("tags", newTags.join(","));
    params.set("sortBy", sortBy);
    params.set("page", "1");
    
    window.history.pushState({}, "", `${window.location.pathname}?${params.toString()}`);
    setSearchParams(params);
    setQuery(newQuery);
    setTags(newTags);
    setPage(1);
    refetch();
  };

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newSortBy = e.target.value;
    const params = new URLSearchParams(searchParams);
    params.set("sortBy", newSortBy);
    
    window.history.pushState({}, "", `${window.location.pathname}?${params.toString()}`);
    setSearchParams(params);
    setSortBy(newSortBy);
    refetch();
  };

  const loadMoreResults = () => {
    setPage(page + 1);
    const params = new URLSearchParams(searchParams);
    params.set("page", (page + 1).toString());
    
    window.history.pushState({}, "", `${window.location.pathname}?${params.toString()}`);
    setSearchParams(params);
    refetch();
  };

  const suggestedFilters = [
    "Machine Learning",
    "CSV",
    "Healthcare",
    "Financial Data",
    "Time Series",
    "Computer Vision",
    "NLP",
    "Public Domain",
    "Images"
  ];

  return (
    <div>
      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow mb-8">
        <div className="p-4 md:p-6 border-b border-neutral-light">
          <h2 className="text-xl font-semibold mb-4">Search Public Datasets</h2>
          <SearchForm 
            onSearch={handleSearch} 
            suggestedFilters={suggestedFilters} 
          />
        </div>
      </div>

      {/* Search Results */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">
            {query || tags.length > 0 ? "Search Results" : "All Datasets"}
          </h2>
          
          <div className="flex items-center">
            <label className="mr-2 text-sm text-neutral-dark">Sort by:</label>
            <select 
              className="border border-neutral-light rounded-md py-1 px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-white"
              value={sortBy}
              onChange={handleSortChange}
            >
              <option value="relevance">Relevance</option>
              <option value="dateAdded">Date Added</option>
              <option value="name">Name</option>
              <option value="quality">Quality Score</option>
            </select>
          </div>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : datasets && datasets.length > 0 ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {datasets.map((dataset) => (
                <DatasetCard key={dataset.id} dataset={dataset} />
              ))}
            </div>
            
            <div className="mt-6 text-center">
              <Button 
                variant="link" 
                className="text-primary font-medium hover:text-primary-dark inline-flex items-center"
                onClick={loadMoreResults}
                disabled={datasets.length < pageSize}
              >
                {datasets.length < pageSize ? "No more results" : "Load more results"}
                <span className="material-icons ml-1">expand_more</span>
              </Button>
            </div>
          </>
        ) : (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <p className="text-lg text-neutral-dark mb-2">No datasets found</p>
            <p className="text-neutral-medium">Try adjusting your search query or filters</p>
          </div>
        )}
      </div>
    </div>
  );
}
