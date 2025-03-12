import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Badge } from "@/components/ui/badge";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardFooter 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Download, 
  ExternalLink, 
  File, 
  Tag, 
  Calendar,
  ArrowLeft,
  Loader2
} from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Dataset, MetadataQuality, ProcessingQueue } from "@shared/schema";
import ProgressBar from "@/components/progress-bar";
import ComplianceChart from "@/components/compliance-chart";
import { formatDistanceToNow } from "date-fns";

interface DatasetDetailsProps {
  id: number;
}

export default function DatasetDetails({ id }: DatasetDetailsProps) {
  const { data, isLoading, error } = useQuery<{
    dataset: Dataset;
    quality?: MetadataQuality;
    processing?: ProcessingQueue;
  }>({
    queryKey: [`/api/datasets/${id}`],
  });

  // Format the date added to a relative time
  const formatDate = (dateString: string | Date) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : dateString;
      return formatDistanceToNow(date, { addSuffix: true });
    } catch (error) {
      return 'Unknown date';
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "processed":
        return <Badge className="bg-green-500">Processed</Badge>;
      case "processing":
        return <Badge className="bg-blue-500">Processing</Badge>;
      case "queued":
        return <Badge className="bg-gray-500">Queued</Badge>;
      case "error":
        return <Badge className="bg-red-500">Error</Badge>;
      default:
        return <Badge className="bg-gray-300">Unknown</Badge>;
    }
  };

  const getQualityBadge = (score?: number) => {
    if (!score) return null;
    
    if (score >= 80) {
      return <Badge className="bg-green-500">High Quality</Badge>;
    } else if (score >= 60) {
      return <Badge className="bg-blue-500">Medium Quality</Badge>;
    } else {
      return <Badge className="bg-yellow-500">Low Quality</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh]">
        <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
        <p className="text-neutral-dark">Loading dataset details...</p>
      </div>
    );
  }

  if (error || !data?.dataset) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh]">
        <AlertCircle className="h-12 w-12 text-destructive mb-4" />
        <h2 className="text-xl font-bold mb-2">Error Loading Dataset</h2>
        <p className="text-neutral-dark mb-4">
          {error instanceof Error ? error.message : "Could not load dataset details"}
        </p>
        <Link href="/search">
          <Button>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Search
          </Button>
        </Link>
      </div>
    );
  }

  const { dataset, quality, processing } = data;

  // Calculate FAIR and schema.org metrics for charts
  const fairMetrics = quality ? [
    { name: "Findable", value: quality.fairFindable || 0 },
    { name: "Accessible", value: quality.fairAccessible || 0 },
    { name: "Interoperable", value: quality.fairInteroperable || 0 },
    { name: "Reusable", value: quality.fairReusable || 0 },
  ] : [];

  const schemaOrgMetrics = quality ? [
    { name: "Required Properties", value: quality.schemaOrgRequired || 0 },
    { name: "Recommended Properties", value: quality.schemaOrgRecommended || 0 },
    { name: "Vocabulary Alignment", value: quality.schemaOrgVocabulary || 0 },
    { name: "Structural Quality", value: quality.schemaOrgStructure || 0 },
  ] : [];

  const fairOverallScore = quality 
    ? Math.round((quality.fairFindable + quality.fairAccessible + quality.fairInteroperable + quality.fairReusable) / 4) 
    : 0;

  const schemaOrgOverallScore = quality 
    ? Math.round((quality.schemaOrgRequired + quality.schemaOrgRecommended + quality.schemaOrgVocabulary + quality.schemaOrgStructure) / 4)
    : 0;

  return (
    <div>
      {/* Back button and dataset title */}
      <div className="mb-6">
        <Link href="/search">
          <Button variant="outline" className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Search
          </Button>
        </Link>
        
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">{dataset.name}</h1>
            <p className="text-neutral-medium mt-1">{dataset.source}</p>
          </div>
          
          <div className="flex items-center gap-2">
            {getStatusBadge(dataset.status)}
            {getQualityBadge(dataset.qualityScore)}
          </div>
        </div>
      </div>

      {/* Main content tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="overview" className="flex-1">Overview</TabsTrigger>
          <TabsTrigger value="metadata" className="flex-1">Metadata</TabsTrigger>
          <TabsTrigger value="quality" className="flex-1">Quality Metrics</TabsTrigger>
          {processing && (
            <TabsTrigger value="processing" className="flex-1">Processing</TabsTrigger>
          )}
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Dataset Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold">Description</h3>
                      <p className="text-neutral-dark mt-1">{dataset.description || "No description available."}</p>
                    </div>
                    
                    {dataset.url && (
                      <div>
                        <h3 className="font-semibold">Source URL</h3>
                        <a 
                          href={dataset.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-primary hover:underline flex items-center mt-1"
                        >
                          {dataset.url}
                          <ExternalLink className="h-4 w-4 ml-1" />
                        </a>
                      </div>
                    )}
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {dataset.format && (
                        <div>
                          <h3 className="font-semibold">Format</h3>
                          <div className="flex items-center mt-1">
                            <File className="h-4 w-4 mr-2 text-neutral-medium" />
                            {dataset.format}
                          </div>
                        </div>
                      )}
                      
                      {dataset.size && (
                        <div>
                          <h3 className="font-semibold">Size</h3>
                          <div className="flex items-center mt-1">
                            <span className="material-icons text-sm mr-2 text-neutral-medium">memory</span>
                            {dataset.size}
                          </div>
                        </div>
                      )}
                      
                      <div>
                        <h3 className="font-semibold">Added</h3>
                        <div className="flex items-center mt-1">
                          <Calendar className="h-4 w-4 mr-2 text-neutral-medium" />
                          {formatDate(dataset.dateAdded)}
                        </div>
                      </div>
                      
                      {dataset.tags && dataset.tags.length > 0 && (
                        <div>
                          <h3 className="font-semibold">Tags</h3>
                          <div className="flex items-center flex-wrap gap-2 mt-2">
                            {dataset.tags.map((tag, index) => (
                              <Badge key={index} variant="outline" className="flex items-center">
                                <Tag className="h-3 w-3 mr-1" />
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex flex-col sm:flex-row sm:justify-between gap-4 border-t pt-4">
                  {dataset.url && (
                    <Button variant="outline" className="w-full sm:w-auto">
                      <Download className="mr-2 h-4 w-4" />
                      Download Original Dataset
                    </Button>
                  )}
                  
                  {dataset.isProcessed && (
                    <Button className="w-full sm:w-auto bg-primary text-white">
                      <Download className="mr-2 h-4 w-4" />
                      Download Processed Metadata
                    </Button>
                  )}
                </CardFooter>
              </Card>
            </div>

            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Quality Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-medium">Overall Quality</h3>
                        <span className="font-bold text-lg">{dataset.qualityScore || 0}%</span>
                      </div>
                      <ProgressBar 
                        value={dataset.qualityScore || 0} 
                        status={dataset.qualityScore && dataset.qualityScore >= 70 ? "completed" : "processing"} 
                        height="h-3"
                      />
                    </div>
                    
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-medium">FAIR Compliance</h3>
                        <span className="font-bold text-lg">{dataset.fairScore || 0}%</span>
                      </div>
                      <ProgressBar 
                        value={dataset.fairScore || 0} 
                        status={dataset.fairScore && dataset.fairScore >= 70 ? "completed" : "processing"} 
                        height="h-3"
                      />
                    </div>
                    
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-medium">Schema.org</h3>
                        <span className="font-bold text-lg">{dataset.schemaOrgScore || 0}%</span>
                      </div>
                      <ProgressBar 
                        value={dataset.schemaOrgScore || 0} 
                        status={dataset.schemaOrgScore && dataset.schemaOrgScore >= 70 ? "completed" : "processing"} 
                        height="h-3"
                      />
                    </div>
                    
                    <Separator />
                    
                    <div className="flex flex-col space-y-2">
                      {dataset.isFairCompliant && (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-5 w-5 mr-2" />
                          <span>FAIR Compliant</span>
                        </div>
                      )}
                      
                      {dataset.schemaOrgScore && dataset.schemaOrgScore >= 70 && (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-5 w-5 mr-2" />
                          <span>Schema.org Compliant</span>
                        </div>
                      )}
                      
                      {dataset.status === "processed" && (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-5 w-5 mr-2" />
                          <span>Processing Complete</span>
                        </div>
                      )}
                      
                      {dataset.status === "processing" && (
                        <div className="flex items-center text-blue-600">
                          <Clock className="h-5 w-5 mr-2" />
                          <span>Processing in Progress</span>
                        </div>
                      )}
                      
                      {dataset.status === "queued" && (
                        <div className="flex items-center text-amber-600">
                          <Clock className="h-5 w-5 mr-2" />
                          <span>Queued for Processing</span>
                        </div>
                      )}
                      
                      {dataset.status === "error" && (
                        <div className="flex items-center text-red-600">
                          <AlertCircle className="h-5 w-5 mr-2" />
                          <span>Processing Error</span>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Metadata Tab */}
        <TabsContent value="metadata">
          <Card>
            <CardHeader>
              <CardTitle>Dataset Metadata</CardTitle>
              <CardDescription>
                Structured metadata following schema.org standards and FAIR principles
              </CardDescription>
            </CardHeader>
            <CardContent>
              {dataset.metadata ? (
                <div className="bg-neutral-lightest p-4 rounded-md overflow-auto max-h-[600px]">
                  <pre className="font-mono text-sm whitespace-pre-wrap">
                    {JSON.stringify(dataset.metadata, null, 2)}
                  </pre>
                </div>
              ) : (
                <div className="bg-neutral-lightest p-8 rounded-md text-center">
                  <Clock className="h-16 w-16 text-neutral-medium mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">Metadata Not Generated Yet</h3>
                  <p className="text-neutral-medium">
                    {dataset.status === "queued" 
                      ? "This dataset is queued for processing." 
                      : dataset.status === "processing" 
                      ? "This dataset is currently being processed." 
                      : "No metadata available for this dataset."}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Quality Metrics Tab */}
        <TabsContent value="quality">
          {quality ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ComplianceChart
                title="FAIR Compliance"
                metrics={fairMetrics}
                overallScore={fairOverallScore}
                color="#3949ab"
              />
              
              <ComplianceChart
                title="Schema.org Compliance"
                metrics={schemaOrgMetrics}
                overallScore={schemaOrgOverallScore}
                color="#00acc1"
              />
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Clock className="h-16 w-16 text-neutral-medium mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">Quality Metrics Not Available</h3>
                <p className="text-neutral-medium">
                  Quality metrics will be generated after the dataset is processed.
                </p>
                
                {dataset.status === "queued" && (
                  <div className="mt-4">
                    <Badge className="bg-neutral-medium">Queued for Processing</Badge>
                  </div>
                )}
                
                {dataset.status === "processing" && (
                  <div className="mt-4">
                    <Badge className="bg-blue-500">Processing in Progress</Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Processing Tab */}
        {processing && (
          <TabsContent value="processing">
            <Card>
              <CardHeader>
                <CardTitle>Processing Status</CardTitle>
                <CardDescription>
                  Current status of dataset processing and metadata generation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">Progress</h3>
                      <Badge className={
                        processing.status === "completed" ? "bg-green-500" : 
                        processing.status === "processing" ? "bg-blue-500" :
                        processing.status === "error" ? "bg-red-500" : "bg-neutral-medium"
                      }>
                        {processing.status}
                      </Badge>
                    </div>
                    <ProgressBar 
                      value={processing.progress || 0} 
                      status={processing.status} 
                      height="h-4"
                    />
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {processing.startTime && (
                      <div>
                        <h3 className="font-medium text-neutral-medium">Started</h3>
                        <p className="mt-1">{new Date(processing.startTime).toLocaleString()}</p>
                      </div>
                    )}
                    
                    {processing.endTime && (
                      <div>
                        <h3 className="font-medium text-neutral-medium">Completed</h3>
                        <p className="mt-1">{new Date(processing.endTime).toLocaleString()}</p>
                      </div>
                    )}
                    
                    {processing.estimatedCompletionTime && !processing.endTime && (
                      <div>
                        <h3 className="font-medium text-neutral-medium">Estimated Completion</h3>
                        <p className="mt-1">{new Date(processing.estimatedCompletionTime).toLocaleString()}</p>
                      </div>
                    )}
                    
                    {processing.error && (
                      <div className="md:col-span-2">
                        <h3 className="font-medium text-neutral-medium">Error</h3>
                        <p className="mt-1 text-red-500">{processing.error}</p>
                      </div>
                    )}
                  </div>
                  
                  {processing.status === "processing" && (
                    <div className="bg-blue-50 p-4 rounded-md">
                      <p className="text-blue-600 flex items-center">
                        <Clock className="h-5 w-5 mr-2" />
                        Processing is ongoing. Please check back later for complete results.
                      </p>
                    </div>
                  )}
                  
                  {processing.status === "error" && (
                    <div className="bg-red-50 p-4 rounded-md">
                      <p className="text-red-600 flex items-center">
                        <AlertCircle className="h-5 w-5 mr-2" />
                        An error occurred during processing. You can try again by requeuing the dataset.
                      </p>
                      <Button variant="outline" className="mt-2">
                        Requeue Dataset
                      </Button>
                    </div>
                  )}
                  
                  {processing.status === "completed" && (
                    <div className="bg-green-50 p-4 rounded-md">
                      <p className="text-green-600 flex items-center">
                        <CheckCircle className="h-5 w-5 mr-2" />
                        Processing completed successfully. Metadata has been generated.
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
