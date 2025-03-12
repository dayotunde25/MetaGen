import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DatasetCard from "@/components/dashboard/dataset-card";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";

export default function DownloadedDatasets() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  
  const { data: allDatasets, isLoading } = useQuery({
    queryKey: ['/api/datasets'],
  });
  
  if (!allDatasets && !isLoading) {
    toast({
      title: "Error",
      description: "Failed to load datasets. Please try again.",
      variant: "destructive",
    });
  }
  
  const processedDatasets = allDatasets?.filter(dataset => dataset.status === 'processed') || [];
  const pendingDatasets = allDatasets?.filter(dataset => ['pending', 'downloading', 'structuring', 'generating'].includes(dataset.status)) || [];
  const failedDatasets = allDatasets?.filter(dataset => dataset.status === 'failed') || [];
  
  const filterDatasets = (datasets) => {
    if (!searchQuery) return datasets;
    
    const query = searchQuery.toLowerCase();
    return datasets.filter(dataset => 
      dataset.title.toLowerCase().includes(query) ||
      dataset.source.toLowerCase().includes(query) ||
      (dataset.description && dataset.description.toLowerCase().includes(query))
    );
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold text-gray-800">Downloaded Datasets</h1>
        
        <div className="relative max-w-xs">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <Input 
            type="text"
            placeholder="Search datasets..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      
      <Tabs defaultValue="all">
        <TabsList>
          <TabsTrigger value="all">
            All Datasets ({allDatasets?.length || 0})
          </TabsTrigger>
          <TabsTrigger value="processed">
            Processed ({processedDatasets.length})
          </TabsTrigger>
          <TabsTrigger value="pending">
            Pending ({pendingDatasets.length})
          </TabsTrigger>
          <TabsTrigger value="failed">
            Failed ({failedDatasets.length})
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="all" className="mt-6">
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
              <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
              <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filterDatasets(allDatasets || []).map((dataset) => (
                <DatasetCard key={dataset.id} dataset={dataset} />
              ))}
              
              {(filterDatasets(allDatasets || []).length === 0) && (
                <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                  <p className="text-gray-500">
                    {searchQuery ? "No datasets match your search criteria." : "No datasets available yet."}
                  </p>
                </div>
              )}
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="processed" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filterDatasets(processedDatasets).map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
            
            {(filterDatasets(processedDatasets).length === 0) && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">
                  {searchQuery ? "No processed datasets match your search criteria." : "No processed datasets available yet."}
                </p>
              </div>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="pending" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filterDatasets(pendingDatasets).map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
            
            {(filterDatasets(pendingDatasets).length === 0) && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">
                  {searchQuery ? "No pending datasets match your search criteria." : "No pending datasets available."}
                </p>
              </div>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="failed" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filterDatasets(failedDatasets).map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
            
            {(filterDatasets(failedDatasets).length === 0) && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">
                  {searchQuery ? "No failed datasets match your search criteria." : "No failed datasets available."}
                </p>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
