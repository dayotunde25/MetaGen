import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { fetchRepositories, searchRepository, addToProcessingQueue } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Globe, Link, Upload, Plus } from "lucide-react";
import { InsertProcessingQueue } from "@shared/schema";

export default function SourceDataset() {
  const [sourceType, setSourceType] = useState<"repository" | "url" | "upload">("repository");
  const [selectedRepository, setSelectedRepository] = useState<string>("kaggle");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const { toast } = useToast();

  // Fetch available repositories
  const { data: repositories = [] } = useQuery({
    queryKey: ['/api/repositories'],
    queryFn: fetchRepositories
  });

  // Repository search mutation
  const { mutate: search, isPending: isSearching } = useMutation({
    mutationFn: async () => {
      if (!searchQuery.trim()) return [];
      return await searchRepository(selectedRepository, searchQuery);
    },
    onSuccess: (data) => {
      setSearchResults(data || []);
    },
    onError: (error) => {
      toast({
        title: "Search failed",
        description: `Failed to search repository: ${error instanceof Error ? error.message : "Unknown error"}`,
        variant: "destructive"
      });
    }
  });

  // Add to processing queue mutation
  const { mutate: addToQueue } = useMutation({
    mutationFn: async (dataset: any) => {
      const queueItem: InsertProcessingQueue = {
        name: dataset.name,
        source: selectedRepository,
        sourceUrl: dataset.url || `https://example.com/${dataset.id}`,
        size: dataset.size || 'Unknown',
        estimatedCompletionTime: '15 minutes'
      };
      return await addToProcessingQueue(queueItem);
    },
    onSuccess: () => {
      toast({
        title: "Dataset added to queue",
        description: "The dataset has been added to processing queue",
        variant: "success"
      });
      queryClient.invalidateQueries({ queryKey: ['/api/processing'] });
    },
    onError: (error) => {
      toast({
        title: "Failed to add dataset",
        description: `Error adding dataset to queue: ${error instanceof Error ? error.message : "Unknown error"}`,
        variant: "destructive"
      });
    }
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    search();
  };

  const handleAddToQueue = (dataset: any) => {
    addToQueue(dataset);
  };

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Source New Dataset</h1>
        <p className="text-slate-600">Find and process public datasets from repositories and generate structured metadata.</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h2 className="text-lg font-semibold mb-4">Dataset Source</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div 
            className={`border rounded-lg p-4 hover:border-primary-400 cursor-pointer transition ${
              sourceType === "repository" ? "bg-primary-50 border-primary-400" : "border-slate-200"
            }`}
            onClick={() => setSourceType("repository")}
          >
            <div className="flex justify-between items-start">
              <div className={`mb-2 ${sourceType === "repository" ? "text-primary-600" : "text-slate-400"}`}>
                <Globe className="h-6 w-6" />
              </div>
              <div className={`h-5 w-5 rounded-full border-2 flex items-center justify-center ${
                sourceType === "repository" ? "border-primary-500" : "border-slate-300"
              }`}>
                {sourceType === "repository" && <div className="h-2 w-2 rounded-full bg-primary-500"></div>}
              </div>
            </div>
            <h3 className="font-medium mb-1">Public Repository</h3>
            <p className="text-sm text-slate-500">Source from known public data repositories.</p>
          </div>
          <div 
            className={`border rounded-lg p-4 hover:border-primary-400 cursor-pointer transition ${
              sourceType === "url" ? "bg-primary-50 border-primary-400" : "border-slate-200"
            }`}
            onClick={() => setSourceType("url")}
          >
            <div className="flex justify-between items-start">
              <div className={`mb-2 ${sourceType === "url" ? "text-primary-600" : "text-slate-400"}`}>
                <Link className="h-6 w-6" />
              </div>
              <div className={`h-5 w-5 rounded-full border-2 flex items-center justify-center ${
                sourceType === "url" ? "border-primary-500" : "border-slate-300"
              }`}>
                {sourceType === "url" && <div className="h-2 w-2 rounded-full bg-primary-500"></div>}
              </div>
            </div>
            <h3 className="font-medium mb-1">URL Source</h3>
            <p className="text-sm text-slate-500">Provide a direct URL to a dataset.</p>
          </div>
          <div 
            className={`border rounded-lg p-4 hover:border-primary-400 cursor-pointer transition ${
              sourceType === "upload" ? "bg-primary-50 border-primary-400" : "border-slate-200"
            }`}
            onClick={() => setSourceType("upload")}
          >
            <div className="flex justify-between items-start">
              <div className={`mb-2 ${sourceType === "upload" ? "text-primary-600" : "text-slate-400"}`}>
                <Upload className="h-6 w-6" />
              </div>
              <div className={`h-5 w-5 rounded-full border-2 flex items-center justify-center ${
                sourceType === "upload" ? "border-primary-500" : "border-slate-300"
              }`}>
                {sourceType === "upload" && <div className="h-2 w-2 rounded-full bg-primary-500"></div>}
              </div>
            </div>
            <h3 className="font-medium mb-1">Upload Files</h3>
            <p className="text-sm text-slate-500">Upload dataset files directly.</p>
          </div>
        </div>

        {sourceType === "repository" && (
          <>
            <div className="mb-6">
              <h3 className="font-medium mb-3">Select Repository</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {repositories.length > 0 ? (
                  repositories.map((repo, index) => (
                    <Button
                      key={index}
                      variant={selectedRepository === repo.toLowerCase() ? "secondary" : "outline"}
                      className={selectedRepository === repo.toLowerCase() ? "border-2 border-primary-500 text-primary-600 font-medium bg-primary-50" : ""}
                      onClick={() => setSelectedRepository(repo.toLowerCase())}
                    >
                      {repo}
                    </Button>
                  ))
                ) : (
                  <>
                    <Button
                      variant={selectedRepository === "kaggle" ? "secondary" : "outline"}
                      className={selectedRepository === "kaggle" ? "border-2 border-primary-500 text-primary-600 font-medium bg-primary-50" : ""}
                      onClick={() => setSelectedRepository("kaggle")}
                    >
                      Kaggle
                    </Button>
                    <Button
                      variant={selectedRepository === "data.gov" ? "secondary" : "outline"}
                      className={selectedRepository === "data.gov" ? "border-2 border-primary-500 text-primary-600 font-medium bg-primary-50" : ""}
                      onClick={() => setSelectedRepository("data.gov")}
                    >
                      Data.gov
                    </Button>
                    <Button
                      variant={selectedRepository === "uci" ? "secondary" : "outline"}
                      className={selectedRepository === "uci" ? "border-2 border-primary-500 text-primary-600 font-medium bg-primary-50" : ""}
                      onClick={() => setSelectedRepository("uci")}
                    >
                      UCI Repository
                    </Button>
                    <Button
                      variant={selectedRepository === "google" ? "secondary" : "outline"}
                      className={selectedRepository === "google" ? "border-2 border-primary-500 text-primary-600 font-medium bg-primary-50" : ""}
                      onClick={() => setSelectedRepository("google")}
                    >
                      Google Dataset Search
                    </Button>
                  </>
                )}
              </div>
            </div>

            <div className="mb-6">
              <div className="flex justify-between items-center mb-3">
                <h3 className="font-medium">Search Repository</h3>
                <div className="text-sm text-slate-500">
                  Showing results for {selectedRepository.charAt(0).toUpperCase() + selectedRepository.slice(1)}
                </div>
              </div>
              <form onSubmit={handleSearch} className="flex">
                <Input
                  type="text"
                  placeholder="Search for datasets..."
                  className="flex-grow rounded-l-lg border border-slate-300 px-4 py-2"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <Button 
                  type="submit" 
                  className="bg-primary-600 text-white px-4 py-2 rounded-r-lg hover:bg-primary-700"
                  disabled={isSearching}
                >
                  {isSearching ? "Searching..." : "Search"}
                </Button>
              </form>
            </div>

            <div>
              <h3 className="font-medium mb-3">Search Results</h3>
              {searchResults.length > 0 ? (
                <div className="border border-slate-200 rounded-lg overflow-hidden">
                  <Table>
                    <TableHeader className="bg-slate-50">
                      <TableRow>
                        <TableHead className="w-1/3">Dataset Name</TableHead>
                        <TableHead>Source</TableHead>
                        <TableHead>Format</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Last Updated</TableHead>
                        <TableHead className="text-right">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {searchResults.map((dataset, index) => (
                        <TableRow key={index}>
                          <TableCell>
                            <div className="flex items-center">
                              <i className="fas fa-table text-slate-400 mr-2"></i>
                              <div>
                                <div className="text-sm font-medium text-slate-900">{dataset.name}</div>
                                <div className="text-xs text-slate-500">{dataset.description?.substring(0, 50)}...</div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="text-sm text-slate-500">{selectedRepository}</TableCell>
                          <TableCell className="text-sm text-slate-500">{dataset.format || "CSV"}</TableCell>
                          <TableCell className="text-sm text-slate-500">{dataset.size || "Unknown"}</TableCell>
                          <TableCell className="text-sm text-slate-500">{dataset.lastUpdated || new Date().toISOString().split('T')[0]}</TableCell>
                          <TableCell className="text-right">
                            <Button 
                              className="inline-flex items-center px-3 py-1 text-sm leading-4 font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700"
                              onClick={() => handleAddToQueue(dataset)}
                            >
                              <Plus className="h-3 w-3 mr-1" />
                              Add
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="border border-slate-200 rounded-lg p-8 text-center">
                  <p className="text-slate-500">
                    {isSearching ? "Searching for datasets..." : 
                      searchQuery ? "No datasets found matching your search criteria." : 
                      "Enter a search term and click Search to find datasets."}
                  </p>
                </div>
              )}
            </div>
          </>
        )}
        
        {sourceType === "url" && (
          <div className="p-4 bg-slate-50 rounded-lg text-center">
            <h3 className="font-medium mb-4">Direct URL Source</h3>
            <div className="mb-4">
              <Input 
                type="text" 
                placeholder="Enter dataset URL (e.g., https://example.com/dataset.csv)" 
                className="w-full mb-2" 
              />
              <p className="text-xs text-slate-500">
                Provide a direct URL to a dataset file. Supported formats: CSV, JSON, XML, Excel
              </p>
            </div>
            <Button className="bg-primary-600 text-white hover:bg-primary-700">
              Add to Processing Queue
            </Button>
          </div>
        )}
        
        {sourceType === "upload" && (
          <div className="p-4 bg-slate-50 rounded-lg text-center">
            <h3 className="font-medium mb-4">Upload Dataset Files</h3>
            <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 mb-4">
              <div className="text-slate-400 mb-2">
                <Upload className="h-8 w-8 mx-auto" />
              </div>
              <p className="mb-4 text-slate-600">Drag and drop files here, or click to browse</p>
              <Button variant="outline">
                Browse Files
              </Button>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Supported formats: CSV, JSON, XML, Excel, Parquet. Maximum file size: 500MB
            </p>
            <Button className="bg-primary-600 text-white hover:bg-primary-700">
              Upload and Process
            </Button>
          </div>
        )}
      </div>
      
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Processing Configuration</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="font-medium mb-3">Metadata Generation</h3>
            <div className="space-y-3">
              <div className="flex items-center">
                <input id="schema-org" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="schema-org" className="ml-2 block text-sm text-slate-700">Schema.org Dataset standard</label>
              </div>
              <div className="flex items-center">
                <input id="dcat" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="dcat" className="ml-2 block text-sm text-slate-700">DCAT (Data Catalog Vocabulary)</label>
              </div>
              <div className="flex items-center">
                <input id="datacite" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" />
                <label htmlFor="datacite" className="ml-2 block text-sm text-slate-700">DataCite Metadata Schema</label>
              </div>
              <div className="flex items-center">
                <input id="dublin-core" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" />
                <label htmlFor="dublin-core" className="ml-2 block text-sm text-slate-700">Dublin Core</label>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="font-medium mb-3">FAIR Compliance</h3>
            <div className="space-y-3">
              <div className="flex items-center">
                <input id="findability" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="findability" className="ml-2 block text-sm text-slate-700">Improve Findability (keywords, identifiers)</label>
              </div>
              <div className="flex items-center">
                <input id="accessibility" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="accessibility" className="ml-2 block text-sm text-slate-700">Enhance Accessibility (access protocols)</label>
              </div>
              <div className="flex items-center">
                <input id="interoperability" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="interoperability" className="ml-2 block text-sm text-slate-700">Ensure Interoperability (standard formats)</label>
              </div>
              <div className="flex items-center">
                <input id="reusability" type="checkbox" className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded" defaultChecked />
                <label htmlFor="reusability" className="ml-2 block text-sm text-slate-700">Optimize Reusability (licenses, provenance)</label>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mb-6">
          <h3 className="font-medium mb-3">Quality Assessment</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center justify-between p-3 border border-slate-200 rounded-lg">
              <label htmlFor="completeness" className="block text-sm font-medium text-slate-700">Completeness</label>
              <select id="completeness" className="rounded-md border-slate-300 text-sm focus:border-primary-500 focus:ring-primary-500">
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
            </div>
            <div className="flex items-center justify-between p-3 border border-slate-200 rounded-lg">
              <label htmlFor="accuracy" className="block text-sm font-medium text-slate-700">Accuracy</label>
              <select id="accuracy" className="rounded-md border-slate-300 text-sm focus:border-primary-500 focus:ring-primary-500">
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
            </div>
            <div className="flex items-center justify-between p-3 border border-slate-200 rounded-lg">
              <label htmlFor="consistency" className="block text-sm font-medium text-slate-700">Consistency</label>
              <select id="consistency" className="rounded-md border-slate-300 text-sm focus:border-primary-500 focus:ring-primary-500">
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="flex justify-end space-x-3">
          <Button variant="outline">Save Configuration</Button>
          <Button>Start Processing</Button>
        </div>
      </div>
    </div>
  );
}
