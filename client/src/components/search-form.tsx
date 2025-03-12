import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import FilterChip from "./filter-chip";

interface SearchFormProps {
  onSearch: (query: string, filters: string[]) => void;
  suggestedFilters?: string[];
}

export default function SearchForm({ onSearch, suggestedFilters = [] }: SearchFormProps) {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeFilters, setActiveFilters] = useState<string[]>([]);

  const handleSearch = () => {
    onSearch(searchQuery, activeFilters);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const addFilter = (filter: string) => {
    if (!activeFilters.includes(filter)) {
      setActiveFilters([...activeFilters, filter]);
    }
  };

  const removeFilter = (filter: string) => {
    setActiveFilters(activeFilters.filter(f => f !== filter));
  };

  return (
    <div>
      <div className="flex flex-col md:flex-row">
        <div className="relative flex-grow mb-4 md:mb-0 md:mr-4">
          <Input
            type="text"
            placeholder="Search by keyword, domain, or format..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyPress}
            className="w-full py-3 px-4 pl-12"
          />
          <Search className="h-5 w-5 absolute left-3 top-3 text-neutral-medium" />
        </div>
        
        <Button 
          className="bg-primary text-white py-3 px-6 flex items-center justify-center hover:bg-primary-dark"
          onClick={handleSearch}
        >
          <Search className="mr-2 h-5 w-5" />
          Search
        </Button>
      </div>
      
      {/* Suggested Filters */}
      {suggestedFilters.length > 0 && (
        <div className="mt-4 flex flex-wrap">
          <span className="text-neutral-medium text-sm mr-2 mt-2">Suggested filters:</span>
          
          {suggestedFilters.map((filter, index) => (
            <FilterChip
              key={index}
              label={filter}
              active={activeFilters.includes(filter)}
              onSelect={() => addFilter(filter)}
            />
          ))}
        </div>
      )}
      
      {/* Active Filters */}
      {activeFilters.length > 0 && (
        <div className="mt-4 flex flex-wrap">
          <span className="text-neutral-medium text-sm mr-2 mt-2">Active filters:</span>
          
          {activeFilters.map((filter, index) => (
            <FilterChip
              key={index}
              label={filter}
              active={true}
              onRemove={() => removeFilter(filter)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
