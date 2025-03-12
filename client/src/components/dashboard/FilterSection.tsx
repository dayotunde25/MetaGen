import { useState } from "react";
import { FilterIcon, XIcon } from "lucide-react";

interface FilterSectionProps {
  onFilterChange: (filters: {
    dataType?: string;
    category?: string;
    sortBy?: string;
    tags?: string[];
  }) => void;
}

const FilterSection: React.FC<FilterSectionProps> = ({ onFilterChange }) => {
  const [filters, setFilters] = useState({
    dataType: "",
    category: "",
    sortBy: "relevance",
    tags: ["FAIR Compliant", "Open Access", "ML Ready"]
  });

  const handleFilterChange = (name: string, value: string) => {
    const updatedFilters = { ...filters, [name]: value };
    setFilters(updatedFilters);
    onFilterChange(updatedFilters);
  };

  const removeTag = (tagToRemove: string) => {
    const updatedTags = filters.tags.filter(tag => tag !== tagToRemove);
    const updatedFilters = { ...filters, tags: updatedTags };
    setFilters(updatedFilters);
    onFilterChange(updatedFilters);
  };

  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
        <div className="bg-gray-50 rounded-lg shadow-sm p-4">
          <div className="flex flex-wrap items-center justify-between">
            <h2 className="text-lg font-medium text-gray-900 mr-4">Filters</h2>
            <div className="flex flex-wrap gap-2 mt-2 sm:mt-0">
              <select 
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                value={filters.dataType}
                onChange={(e) => handleFilterChange("dataType", e.target.value)}
              >
                <option value="">All Data Types</option>
                <option value="Tabular">Tabular</option>
                <option value="Image">Image</option>
                <option value="Text">Text</option>
                <option value="Time Series">Time Series</option>
                <option value="Graph">Graph</option>
                <option value="Audio">Audio</option>
              </select>
              
              <select 
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                value={filters.category}
                onChange={(e) => handleFilterChange("category", e.target.value)}
              >
                <option value="">All Categories</option>
                <option value="Healthcare">Healthcare</option>
                <option value="Financial">Financial</option>
                <option value="Climate">Climate</option>
                <option value="Social Science">Social Science</option>
                <option value="Computer Vision">Computer Vision</option>
                <option value="NLP">NLP</option>
              </select>
              
              <select 
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                value={filters.sortBy}
                onChange={(e) => handleFilterChange("sortBy", e.target.value)}
              >
                <option value="relevance">Sort By: Relevance</option>
                <option value="recent">Most Recent</option>
                <option value="downloaded">Most Downloaded</option>
                <option value="quality">Highest Quality</option>
                <option value="size_asc">Size (Ascending)</option>
                <option value="size_desc">Size (Descending)</option>
              </select>
              
              <button 
                type="button" 
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                <FilterIcon className="mr-2 h-4 w-4" />
                More Filters
              </button>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {filters.tags.map((tag, index) => {
              let bgColor = 'bg-primary-100 text-primary-800';
              let hoverBg = 'hover:bg-primary-200 hover:text-primary-500';
              let textColor = 'text-primary-400';
              
              if (tag === 'Open Access') {
                bgColor = 'bg-green-100 text-green-800';
                hoverBg = 'hover:bg-green-200 hover:text-green-500';
                textColor = 'text-green-400';
              } else if (tag === 'ML Ready') {
                bgColor = 'bg-blue-100 text-blue-800';
                hoverBg = 'hover:bg-blue-200 hover:text-blue-500';
                textColor = 'text-blue-400';
              }

              return (
                <span key={index} className={`inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium ${bgColor}`}>
                  {tag}
                  <button 
                    type="button" 
                    className={`ml-1.5 inline-flex flex-shrink-0 h-4 w-4 rounded-full items-center justify-center ${textColor} ${hoverBg} focus:outline-none`}
                    onClick={() => removeTag(tag)}
                  >
                    <XIcon className="h-3 w-3" />
                  </button>
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterSection;
