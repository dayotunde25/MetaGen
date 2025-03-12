import { useState } from "react";
import { useNavigate } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Search, Filter, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";

export default function SearchSection() {
  const [, navigate] = useNavigate();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [dataFormat, setDataFormat] = useState("");
  const [category, setCategory] = useState("");
  const [dateRange, setDateRange] = useState("");
  
  const handleSearch = () => {
    // Construct query parameters
    const params = new URLSearchParams();
    if (searchQuery) params.append("q", searchQuery);
    if (dataFormat) params.append("format", dataFormat);
    if (category) params.append("category", category);
    if (dateRange) params.append("dateRange", dateRange);
    
    // Navigate to search page with query parameters
    navigate(`/search?${params.toString()}`);
  };
  
  const handleReset = () => {
    setSearchQuery("");
    setDataFormat("");
    setCategory("");
    setDateRange("");
  };
  
  return (
    <Card className="bg-white shadow rounded-lg">
      <CardContent className="p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Search for Datasets</h2>
        <div className="relative">
          <div className="flex rounded-md shadow-sm">
            <div className="relative flex-grow focus-within:z-10">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <Input 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for datasets using semantic search..." 
                className="pl-10"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSearch();
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
            <button 
              onClick={() => setShowAdvanced(!showAdvanced)} 
              className="text-sm text-primary-600 font-medium flex items-center mb-3"
            >
              <span>Advanced Search Options</span>
              {showAdvanced ? (
                <ChevronUp className="ml-1 h-4 w-4" />
              ) : (
                <ChevronDown className="ml-1 h-4 w-4" />
              )}
            </button>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
          <Button onClick={handleSearch}>
            <Search className="h-4 w-4 mr-1" />
            Search
          </Button>
          <Button variant="outline" onClick={handleReset}>
            <RefreshCw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
