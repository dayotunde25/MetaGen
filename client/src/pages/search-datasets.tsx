import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import DatasetCard from "@/components/dashboard/dataset-card";
import { Search, Filter, RefreshCw } from "lucide-react";

export default function SearchDatasets() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [dataFormat, setDataFormat] = useState("");
  const [category, setCategory] = useState("");
  const [dateRange, setDateRange] = useState("");
  
  const { data: searchResults, isLoading, refetch } = useQuery({
    queryKey: ['/api/datasets/search', searchQuery, dataFormat, category, dateRange],
    enabled: false,
  });
  
  const performSearch = () => {
    const queryParams = new URLSearchParams();
    if (searchQuery) queryParams.append('q', searchQuery);
    if (dataFormat) queryParams.append('format', dataFormat);
    if (category) queryParams.append('category', category);
    if (dateRange) queryParams.append('dateRange', dateRange);
    
    refetch();
  };
  
  const resetSearch = () => {
    setSearchQuery("");
    setDataFormat("");
    setCategory("");
    setDateRange("");
    // Clear search results by not triggering a search
  };
  
  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Search for Datasets</h2>
        <div className="relative">
          <div className="flex rounded-md shadow-sm">
            <div className="relative flex-grow focus-within:z-10">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <Input 
                type="text" 
                placeholder="Search for datasets using semantic search..." 
                className="pl-10"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') performSearch();
                }}
              />
            </div>
            <Button 
              variant="outline" 
              className="ml-2 flex items-center" 
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
          </div>
        </div>
        
        {/* Advanced Search Options */}
        {showAdvanced && (
          <div className="mt-4">
            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="data-format">Data Format</Label>
                <Select value={dataFormat} onValueChange={setDataFormat}>
                  <SelectTrigger id="data-format" className="mt-1">
                    <SelectValue placeholder="Any Format" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Any Format</SelectItem>
                    <SelectItem value="CSV">CSV</SelectItem>
                    <SelectItem value="JSON">JSON</SelectItem>
                    <SelectItem value="XML">XML</SelectItem>
                    <SelectItem value="Parquet">Parquet</SelectItem>
                    <SelectItem value="Avro">Avro</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="category">Category</Label>
                <Select value={category} onValueChange={setCategory}>
                  <SelectTrigger id="category" className="mt-1">
                    <SelectValue placeholder="All Categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Categories</SelectItem>
                    <SelectItem value="Healthcare">Healthcare</SelectItem>
                    <SelectItem value="Climate">Climate & Environment</SelectItem>
                    <SelectItem value="Economics">Economics</SelectItem>
                    <SelectItem value="Social Sciences">Social Sciences</SelectItem>
                    <SelectItem value="Physics">Physics</SelectItem>
                    <SelectItem value="Earth Science">Earth Science</SelectItem>
                    <SelectItem value="Genomics">Genomics</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="date-range">Date Range</Label>
                <Select value={dateRange} onValueChange={setDateRange}>
                  <SelectTrigger id="date-range" className="mt-1">
                    <SelectValue placeholder="Any Time" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Any Time</SelectItem>
                    <SelectItem value="past-day">Past 24 Hours</SelectItem>
                    <SelectItem value="past-week">Past Week</SelectItem>
                    <SelectItem value="past-month">Past Month</SelectItem>
                    <SelectItem value="past-year">Past Year</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        )}
        
        {/* Search Buttons */}
        <div className="mt-4 flex flex-col sm:flex-row sm:justify-end space-y-2 sm:space-y-0 sm:space-x-2">
          <Button onClick={performSearch}>
            <Search className="h-4 w-4 mr-1" />
            Search
          </Button>
          <Button variant="outline" onClick={resetSearch}>
            <RefreshCw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>
      </div>
      
      {/* Search Results */}
      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Search Results</h2>
        
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {searchResults?.map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
            
            {searchResults && searchResults.length === 0 && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">No datasets found matching your search criteria.</p>
              </div>
            )}
            
            {!searchResults && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">Use the search box above to find datasets.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
