import { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { useSearch } from "@/hooks/useSearch";
import { SearchIcon, SlidersIcon, ChevronDownIcon } from "lucide-react";

const Hero = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { toast } = useToast();
  const { search, isSearching } = useSearch();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) {
      toast({
        title: "Search query is empty",
        description: "Please enter a search term",
        variant: "destructive"
      });
      return;
    }
    
    try {
      await search(searchQuery);
    } catch (error) {
      toast({
        title: "Search failed",
        description: "There was an error processing your search",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="bg-primary-700 text-white">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-3xl font-extrabold sm:text-4xl">
            Generate High-Quality Metadata for AI Research Datasets
          </h1>
          <p className="mt-4 text-lg">
            Discover, download, and structure datasets according to schema.org and FAIR principles to enhance AI research initiatives.
          </p>
        </div>
        
        <div className="mt-10 max-w-xl mx-auto">
          <form onSubmit={handleSearch}>
            <div className="relative rounded-md shadow-lg">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <SearchIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                className="block w-full pl-10 pr-12 py-3 border-0 rounded-md text-gray-900 placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                placeholder="Search for datasets (e.g., 'climate change time series', 'sentiment analysis corpus')"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={isSearching}
              />
              <div className="absolute inset-y-0 right-0 flex py-1.5 pr-1.5">
                <button 
                  type="button" 
                  className="inline-flex items-center border border-transparent rounded px-2 text-sm font-medium text-primary-600 hover:text-primary-700"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  <SlidersIcon className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="mt-2 flex justify-between text-xs text-gray-100">
              <span>Try: "machine learning benchmark", "healthcare data", "financial time series"</span>
              <button 
                type="button" 
                className="hover:text-white flex items-center"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <span>Advanced search</span>
                <ChevronDownIcon className="ml-1 h-4 w-4" />
              </button>
            </div>
            
            {showAdvanced && (
              <div className="mt-3 bg-white p-4 rounded-md shadow text-gray-800">
                <h3 className="font-medium text-sm mb-2">Advanced Search Options</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Data Type</label>
                    <select className="w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500">
                      <option value="">Any Type</option>
                      <option value="tabular">Tabular</option>
                      <option value="image">Image</option>
                      <option value="text">Text</option>
                      <option value="timeseries">Time Series</option>
                      <option value="audio">Audio</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Category</label>
                    <select className="w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500">
                      <option value="">Any Category</option>
                      <option value="healthcare">Healthcare</option>
                      <option value="finance">Finance</option>
                      <option value="climate">Climate</option>
                      <option value="nlp">NLP</option>
                      <option value="computer-vision">Computer Vision</option>
                    </select>
                  </div>
                </div>
                <div className="flex justify-end mt-3">
                  <button
                    type="submit"
                    className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Search
                  </button>
                </div>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default Hero;
