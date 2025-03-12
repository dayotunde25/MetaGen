import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import MetadataPreviewModal from "@/components/metadata/metadata-preview-modal";
import { Search, FileSearch, Download, Check } from "lucide-react";

export default function GeneratedMetadata() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [showModal, setShowModal] = useState(false);
  
  const { data: allDatasets, isLoading } = useQuery({
    queryKey: ['/api/datasets'],
  });
  
  // Only get processed datasets with metadata
  const processedDatasets = allDatasets?.filter(dataset => dataset.status === 'processed') || [];
  
  const filterDatasets = (datasets) => {
    if (!searchQuery) return datasets;
    
    const query = searchQuery.toLowerCase();
    return datasets.filter(dataset => 
      dataset.title.toLowerCase().includes(query) ||
      dataset.source.toLowerCase().includes(query) ||
      (dataset.description && dataset.description.toLowerCase().includes(query))
    );
  };
  
  const handleViewMetadata = (datasetId: number) => {
    setSelectedDatasetId(datasetId);
    setShowModal(true);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold text-gray-800">Generated Metadata</h1>
        
        <div className="relative max-w-xs">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <Input 
            type="text"
            placeholder="Search metadata..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      
      {isLoading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="h-40 bg-white shadow rounded-lg animate-pulse"></div>
          <div className="h-40 bg-white shadow rounded-lg animate-pulse"></div>
          <div className="h-40 bg-white shadow rounded-lg animate-pulse"></div>
          <div className="h-40 bg-white shadow rounded-lg animate-pulse"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {filterDatasets(processedDatasets).map((dataset) => (
            <Card key={dataset.id} className="overflow-hidden">
              <CardContent className="p-0">
                <div className="p-5 border-b border-gray-200">
                  <div className="flex justify-between items-start">
                    <div>
                      <Badge variant="outline" className="mb-1">
                        {dataset.category}
                      </Badge>
                      <h3 className="text-base font-semibold text-gray-900 leading-tight">
                        {dataset.title}
                      </h3>
                      <p className="text-sm text-gray-500 mt-1">{dataset.source}</p>
                    </div>
                    <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
                      <Check className="h-3 w-3 mr-1" />
                      Processed
                    </Badge>
                  </div>
                  
                  <div className="mt-3 text-sm text-gray-600 line-clamp-2">
                    <p>{dataset.description}</p>
                  </div>
                  
                  <div className="mt-3 flex flex-wrap gap-1">
                    {dataset.formats?.map((format, idx) => (
                      <Badge key={idx} variant="secondary" className="bg-blue-50 text-blue-700 hover:bg-blue-50">
                        {format}
                      </Badge>
                    ))}
                    <Badge variant="secondary" className="bg-purple-50 text-purple-700 hover:bg-purple-50">
                      FAIR Compliant
                    </Badge>
                  </div>
                </div>
                
                <div className="px-5 py-3 bg-gray-50 flex justify-between items-center">
                  <span className="text-xs text-gray-500">Size: {dataset.size}</span>
                  <div className="flex space-x-2">
                    <Button size="sm" variant="ghost" onClick={() => handleViewMetadata(dataset.id)}>
                      <FileSearch className="h-4 w-4" />
                    </Button>
                    <Button size="sm" variant="ghost">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
          
          {(filterDatasets(processedDatasets).length === 0) && (
            <div className="col-span-2 bg-white shadow rounded-lg p-6 text-center">
              <p className="text-gray-500">
                {searchQuery ? "No metadata records match your search criteria." : "No metadata records available yet."}
              </p>
            </div>
          )}
        </div>
      )}
      
      {showModal && selectedDatasetId && (
        <MetadataPreviewModal 
          datasetId={selectedDatasetId} 
          isOpen={showModal} 
          onClose={() => setShowModal(false)} 
        />
      )}
    </div>
  );
}
