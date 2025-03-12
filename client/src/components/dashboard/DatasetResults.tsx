import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import DatasetCard from "./DatasetCard";
import MetadataModal from "./MetadataModal";
import { useSearch } from "@/hooks/useSearch";
import { Dataset, Metadata } from "@shared/schema";

const DatasetResults = () => {
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [showMetadataModal, setShowMetadataModal] = useState(false);
  const { searchResults, isSearching } = useSearch();

  const { data: datasetsData, isLoading } = useQuery<{ datasets: Dataset[] }>({
    queryKey: ['/api/datasets'],
    enabled: !searchResults, // Only fetch all datasets if not searching
  });

  const { data: metadataData, isLoading: isLoadingMetadata } = useQuery<{ metadata: Metadata }>({
    queryKey: ['/api/datasets', selectedDataset?.id, 'metadata'],
    enabled: !!selectedDataset,
  });

  useEffect(() => {
    if (metadataData && !isLoadingMetadata) {
      setMetadata(metadataData.metadata);
    }
  }, [metadataData, isLoadingMetadata]);

  const handleViewMetadata = async (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setShowMetadataModal(true);
  };

  const datasets = searchResults 
    ? searchResults.results.map(result => result.dataset)
    : datasetsData?.datasets || [];

  const isLoaded = !isLoading && !isSearching;
  const hasResults = datasets.length > 0;

  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          {searchResults 
            ? `Search Results for "${searchResults.query}"` 
            : "Available Datasets"}
        </h2>
        
        {/* Loading state */}
        {(isLoading || isSearching) && (
          <div className="space-y-6">
            {[1, 2].map((i) => (
              <div key={i} className="border rounded-lg">
                <div className="p-4 border-b">
                  <Skeleton className="h-6 w-3/4 mb-2" />
                  <Skeleton className="h-4 w-1/2" />
                </div>
                <div className="p-4 space-y-4">
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-1/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-1/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-1/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-4 rounded-b-lg flex justify-between">
                  <Skeleton className="h-4 w-1/4" />
                  <div className="flex space-x-3">
                    <Skeleton className="h-8 w-20" />
                    <Skeleton className="h-8 w-20" />
                    <Skeleton className="h-8 w-20" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {isLoaded && !hasResults && (
          <div className="text-center py-12 border rounded-lg bg-gray-50">
            <h3 className="mt-2 text-sm font-medium text-gray-900">No datasets found</h3>
            <p className="mt-1 text-sm text-gray-500">
              Try adjusting your search or filters to find what you're looking for.
            </p>
          </div>
        )}
        
        {/* Results */}
        {isLoaded && hasResults && (
          <div>
            {datasets.map((dataset) => (
              <DatasetCard 
                key={dataset.id} 
                dataset={dataset} 
                onViewMetadata={handleViewMetadata}
              />
            ))}
            
            <div className="mt-8">
              <nav className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6" aria-label="Pagination">
                <div className="hidden sm:block">
                  <p className="text-sm text-gray-700">
                    Showing <span className="font-medium">1</span> to <span className="font-medium">{datasets.length}</span> of <span className="font-medium">{datasets.length}</span> results
                  </p>
                </div>
                <div className="flex-1 flex justify-between sm:justify-end">
                  <button
                    disabled
                    className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Previous
                  </button>
                  <button
                    disabled
                    className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Next
                  </button>
                </div>
              </nav>
            </div>
          </div>
        )}
      </div>
      
      {showMetadataModal && selectedDataset && (
        <MetadataModal
          isOpen={showMetadataModal}
          onClose={() => setShowMetadataModal(false)}
          dataset={selectedDataset}
          metadata={metadata}
          isLoading={isLoadingMetadata}
        />
      )}
    </div>
  );
};

export default DatasetResults;
