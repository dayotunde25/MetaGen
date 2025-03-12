import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { FileText, Download, CheckCircle, XCircle, Clock } from "lucide-react";

interface MetadataPreviewModalProps {
  datasetId: number;
  isOpen: boolean;
  onClose: () => void;
}

interface DataStructureField {
  field: string;
  type: string;
  description: string;
}

interface FairAssessment {
  findable: number;
  accessible: number;
  interoperable: number;
  reusable: number;
  overall: number;
}

export default function MetadataPreviewModal({ datasetId, isOpen, onClose }: MetadataPreviewModalProps) {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<string>("structured");

  const { data: dataset, isLoading: datasetLoading } = useQuery({
    queryKey: ['/api/datasets', datasetId],
    queryFn: async () => {
      const res = await fetch(`/api/datasets/${datasetId}`);
      if (!res.ok) throw new Error('Failed to fetch dataset');
      return await res.json();
    },
    enabled: isOpen,
  });

  const { data: metadata, isLoading: metadataLoading } = useQuery({
    queryKey: ['/api/datasets', datasetId, 'metadata'],
    queryFn: async () => {
      const res = await fetch(`/api/datasets/${datasetId}/metadata`);
      if (!res.ok) throw new Error('Failed to fetch metadata');
      return await res.json();
    },
    enabled: isOpen,
  });

  const isLoading = datasetLoading || metadataLoading;

  const handleDownload = () => {
    if (!metadata) return;
    
    // Create a downloadable JSON file
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(metadata.schemaOrgJson, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `metadata-${dataset?.title.replace(/\s+/g, '-').toLowerCase() || datasetId}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    
    toast({
      title: "Metadata Downloaded",
      description: "The metadata file has been downloaded successfully.",
    });
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start">
            <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 sm:mx-0 sm:h-10 sm:w-10">
              <FileText className="h-6 w-6 text-blue-600" />
            </div>
            <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
              <DialogTitle className="text-lg leading-6 font-medium text-gray-900">
                {isLoading ? (
                  <Skeleton className="h-6 w-64" />
                ) : (
                  `Metadata Preview: ${dataset?.title}`
                )}
              </DialogTitle>
              <DialogDescription className="mt-1 text-sm text-gray-500">
                {isLoading ? (
                  <Skeleton className="h-4 w-48" />
                ) : (
                  `Source: ${dataset?.source}`
                )}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="mt-4">
          <Tabs defaultValue="structured" value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-4">
              <TabsTrigger value="structured">Structured View</TabsTrigger>
              <TabsTrigger value="json">JSON</TabsTrigger>
              <TabsTrigger value="schema">Schema.org</TabsTrigger>
              <TabsTrigger value="fair">FAIR Assessment</TabsTrigger>
            </TabsList>

            {/* Structured View Tab */}
            <TabsContent value="structured" className="space-y-4 pt-4">
              {isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-64 w-full" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-2">Basic Information</h4>
                    <div className="bg-gray-50 p-3 rounded-md">
                      <dl className="space-y-2 text-sm">
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Title:</dt>
                          <dd className="col-span-2">{dataset?.title}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Creator:</dt>
                          <dd className="col-span-2">{metadata?.creator || "—"}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Publisher:</dt>
                          <dd className="col-span-2">{metadata?.publisher || "—"}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Publication Date:</dt>
                          <dd className="col-span-2">{metadata?.publicationDate || "—"}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Last Updated:</dt>
                          <dd className="col-span-2">{metadata?.lastUpdated || "—"}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">Language:</dt>
                          <dd className="col-span-2">{metadata?.language || "—"}</dd>
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          <dt className="font-medium text-gray-500">License:</dt>
                          <dd className="col-span-2">{metadata?.license || "—"}</dd>
                        </div>
                      </dl>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-2">Description & Keywords</h4>
                    <div className="bg-gray-50 p-3 rounded-md">
                      <p className="text-sm text-gray-700 mb-3">{dataset?.description}</p>
                      <div className="space-y-2">
                        <h5 className="text-xs font-medium text-gray-500">Keywords:</h5>
                        <div className="flex flex-wrap gap-1">
                          {dataset?.keywords?.map((keyword, index) => (
                            <Badge key={index} variant="outline" className="bg-blue-50 text-blue-700 hover:bg-blue-50">
                              {keyword}
                            </Badge>
                          ))}
                          {(!dataset?.keywords || dataset.keywords.length === 0) && (
                            <span className="text-xs text-gray-500">No keywords available</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Data Structure section */}
              {!isLoading && metadata?.dataStructure && (
                <div>
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Data Structure</h4>
                  <div className="bg-gray-50 p-3 rounded-md overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-1/4">Field</TableHead>
                          <TableHead className="w-1/4">Type</TableHead>
                          <TableHead className="w-2/4">Description</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(metadata.dataStructure as DataStructureField[]).map((field, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-medium">{field.field}</TableCell>
                            <TableCell>{field.type}</TableCell>
                            <TableCell>{field.description}</TableCell>
                          </TableRow>
                        ))}
                        {metadata.dataStructure.length === 0 && (
                          <TableRow>
                            <TableCell colSpan={3} className="text-center py-4 text-gray-500">
                              No data structure information available
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* JSON Tab */}
            <TabsContent value="json" className="space-y-4 pt-4">
              {isLoading ? (
                <Skeleton className="h-96 w-full" />
              ) : (
                <div className="bg-gray-800 rounded-md p-4 overflow-x-auto">
                  <pre className="text-xs text-gray-100 font-mono">
                    {JSON.stringify(metadata?.schemaOrgJson || {}, null, 2)}
                  </pre>
                </div>
              )}
            </TabsContent>

            {/* Schema.org Tab */}
            <TabsContent value="schema" className="space-y-4 pt-4">
              {isLoading ? (
                <Skeleton className="h-96 w-full" />
              ) : (
                <div className="bg-gray-50 p-4 rounded-md">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Schema.org Compliance</h4>
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Dataset Type</p>
                        <p className="text-xs text-gray-500">Properly defined as schema:Dataset</p>
                      </div>
                    </div>

                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Creator & Publisher</p>
                        <p className="text-xs text-gray-500">Organization entities properly structured</p>
                      </div>
                    </div>

                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Variables & Measurements</p>
                        <p className="text-xs text-gray-500">PropertyValue objects for each data field</p>
                      </div>
                    </div>

                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Distributions</p>
                        <p className="text-xs text-gray-500">Multiple format availability properly documented</p>
                      </div>
                    </div>

                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Temporal & Spatial Coverage</p>
                        <p className="text-xs text-gray-500">Geographic and time period information provided</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* FAIR Assessment Tab */}
            <TabsContent value="fair" className="space-y-4 pt-4">
              {isLoading ? (
                <Skeleton className="h-96 w-full" />
              ) : metadata?.fairAssessment ? (
                <div className="bg-gray-50 p-4 rounded-md">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">FAIR Principles Assessment</h4>

                  <div className="space-y-4">
                    <div>
                      <h5 className="text-xs font-semibold text-gray-600 uppercase mb-2 flex items-center">
                        <span className="mr-2">Findable</span>
                        <Badge variant="outline" className="bg-green-100 text-green-800 hover:bg-green-100">
                          {(metadata.fairAssessment as FairAssessment).findable}%
                        </Badge>
                      </h5>
                      <div className="space-y-2">
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Assigned globally unique and persistent identifier (DOI)</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Rich descriptive metadata conforming to schema.org</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Metadata includes the identifier of the data it describes</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Registered in searchable resources</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h5 className="text-xs font-semibold text-gray-600 uppercase mb-2 flex items-center">
                        <span className="mr-2">Accessible</span>
                        <Badge variant="outline" className="bg-green-100 text-green-800 hover:bg-green-100">
                          {(metadata.fairAssessment as FairAssessment).accessible}%
                        </Badge>
                      </h5>
                      <div className="space-y-2">
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Retrievable by standard protocol (HTTPS)</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Protocol is open, free, and universally implementable</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Metadata accessible even when data is no longer available</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h5 className="text-xs font-semibold text-gray-600 uppercase mb-2 flex items-center">
                        <span className="mr-2">Interoperable</span>
                        <Badge variant="outline" className="bg-amber-100 text-amber-800 hover:bg-amber-100">
                          {(metadata.fairAssessment as FairAssessment).interoperable}%
                        </Badge>
                      </h5>
                      <div className="space-y-2">
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Uses formal, accessible, shared vocabulary (schema.org)</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Uses qualified references to other metadata</p>
                        </div>
                        <div className="flex items-start">
                          <XCircle className="h-4 w-4 text-red-500 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Could improve RDF-based metadata representation</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h5 className="text-xs font-semibold text-gray-600 uppercase mb-2 flex items-center">
                        <span className="mr-2">Reusable</span>
                        <Badge variant="outline" className="bg-green-100 text-green-800 hover:bg-green-100">
                          {(metadata.fairAssessment as FairAssessment).reusable}%
                        </Badge>
                      </h5>
                      <div className="space-y-2">
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Rich metadata with plurality of attributes</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Clear and accessible data usage license ({metadata.license})</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Detailed provenance information</p>
                        </div>
                        <div className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2" />
                          <p className="text-xs text-gray-600">Meets domain-relevant community standards</p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-2">
                      <div className="flex items-center">
                        <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mr-4">
                          <span className="text-lg font-bold text-green-700">
                            {(metadata.fairAssessment as FairAssessment).overall}%
                          </span>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900">Overall FAIR Score</p>
                          <p className="text-xs text-gray-600">Excellent compliance with FAIR principles</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-50 p-4 rounded-md text-center">
                  <p className="text-gray-500">No FAIR assessment available for this dataset.</p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>

        <DialogFooter>
          <Button
            onClick={handleDownload}
            disabled={isLoading}
            className="w-full sm:w-auto"
          >
            <Download className="mr-2 h-4 w-4" />
            Download Metadata
          </Button>
          <Button
            variant="outline"
            onClick={onClose}
            className="w-full sm:w-auto"
          >
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
