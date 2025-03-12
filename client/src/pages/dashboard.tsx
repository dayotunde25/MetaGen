import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import StatsCard from "@/components/stats-card";
import DatasetCard from "@/components/dataset-card";
import ProcessingItem from "@/components/processing-item";
import ComplianceChart from "@/components/compliance-chart";
import SearchForm from "@/components/search-form";
import { Dataset, ProcessingQueue, StatsData } from "@shared/schema";

export default function Dashboard() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [processingQueue, setProcessingQueue] = useState<(ProcessingQueue & { dataset: Dataset })[]>([]);

  // Fetch stats
  const { data: stats, isLoading: isLoadingStats } = useQuery<StatsData>({
    queryKey: ["/api/stats"],
  });

  // Fetch recent datasets
  const { data: recentDatasets, isLoading: isLoadingDatasets } = useQuery<Dataset[]>({
    queryKey: ["/api/datasets"],
  });

  // Fetch processing queue
  const { data: queue, isLoading: isLoadingQueue } = useQuery<ProcessingQueue[]>({
    queryKey: ["/api/processing-queue"],
  });

  useEffect(() => {
    if (recentDatasets) {
      // Sort by date added and get the 3 most recent
      const sorted = [...recentDatasets].sort((a, b) => {
        const dateA = a.dateAdded instanceof Date ? a.dateAdded : new Date(a.dateAdded);
        const dateB = b.dateAdded instanceof Date ? b.dateAdded : new Date(b.dateAdded);
        return dateB.getTime() - dateA.getTime();
      });
      setDatasets(sorted.slice(0, 3));
    }
  }, [recentDatasets]);

  useEffect(() => {
    if (queue && recentDatasets) {
      // Join processing queue with dataset details
      const queueWithDatasets = queue
        .filter(q => q.status === "processing" || q.status === "queued")
        .map(q => {
          const dataset = recentDatasets.find(d => d.id === q.datasetId);
          if (dataset) {
            return { ...q, dataset };
          }
          return null;
        })
        .filter(Boolean) as (ProcessingQueue & { dataset: Dataset })[];
      
      setProcessingQueue(queueWithDatasets.slice(0, 2));
    }
  }, [queue, recentDatasets]);

  const handleSearch = (query: string, filters: string[]) => {
    // Navigate to search page with query parameters
    window.location.href = `/search?q=${encodeURIComponent(query)}&tags=${filters.join(',')}`;
  };

  // FAIR compliance sample data for charts
  const fairMetrics = [
    { name: "Findable", value: 86 },
    { name: "Accessible", value: 92 },
    { name: "Interoperable", value: 74 },
    { name: "Reusable", value: 88 },
  ];

  // Schema.org compliance sample data for charts
  const schemaOrgMetrics = [
    { name: "Required Properties", value: 94 },
    { name: "Recommended Properties", value: 82 },
    { name: "Vocabulary Alignment", value: 68 },
    { name: "Structural Quality", value: 90 },
  ];

  const suggestedFilters = [
    "Machine Learning",
    "CSV",
    "Healthcare",
    "Financial Data",
    "Recently Added",
  ];

  return (
    <div>
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="Total Datasets"
          value={stats?.totalDatasets ?? 0}
          icon="dataset"
          change={{ value: 12, isPositive: true }}
          color="primary"
        />
        
        <StatsCard
          title="Processed Datasets"
          value={stats?.processedDatasets ?? 0}
          icon="verified"
          change={{ value: 8, isPositive: true }}
          color="secondary"
        />
        
        <StatsCard
          title="FAIR-Compliant"
          value={stats?.fairCompliantDatasets ?? 0}
          icon="thumb_up"
          change={{ value: 15, isPositive: true }}
          color="accent"
        />
        
        <StatsCard
          title="Processing Queue"
          value={stats?.queuedDatasets ?? 0}
          icon="pending"
          change={{ value: 3, isPositive: false }}
          color="info"
        />
      </div>

      {/* Search and Discovery */}
      <div className="bg-white rounded-lg shadow mb-8">
        <div className="p-4 md:p-6 border-b border-neutral-light">
          <h2 className="text-xl font-semibold mb-4">Search Public Datasets</h2>
          <SearchForm 
            onSearch={handleSearch} 
            suggestedFilters={suggestedFilters} 
          />
        </div>
      </div>

      {/* Recently Added Datasets */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Recently Added Datasets</h2>
          
          <div className="flex items-center">
            <label className="mr-2 text-sm text-neutral-dark">Sort by:</label>
            <select className="border border-neutral-light rounded-md py-1 px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-white">
              <option value="relevance">Relevance</option>
              <option value="date" selected>Date Added</option>
              <option value="name">Name</option>
              <option value="quality">Quality Score</option>
            </select>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoadingDatasets ? (
            Array(3).fill(0).map((_, i) => (
              <div key={i} className="bg-white rounded-lg shadow h-64 animate-pulse"></div>
            ))
          ) : datasets.length > 0 ? (
            datasets.map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))
          ) : (
            <div className="col-span-3 text-center py-8 bg-white rounded-lg shadow">
              <p className="text-neutral-medium">No datasets found. Add one to get started.</p>
            </div>
          )}
        </div>
        
        <div className="mt-6 text-center">
          <Link href="/search">
            <Button variant="link" className="text-primary font-medium hover:text-primary-dark inline-flex items-center">
              View all datasets
              <span className="material-icons ml-1">expand_more</span>
            </Button>
          </Link>
        </div>
      </div>

      {/* Data Processing Section */}
      <div className="bg-white rounded-lg shadow mb-8">
        <div className="p-4 md:p-6 border-b border-neutral-light">
          <h2 className="text-xl font-semibold mb-2">Data Processing Queue</h2>
          <p className="text-neutral-medium text-sm mb-4">Datasets being downloaded and processed to generate metadata</p>
          
          <div className="space-y-4">
            {isLoadingQueue ? (
              Array(2).fill(0).map((_, i) => (
                <div key={i} className="border border-neutral-light rounded-lg p-4 h-32 animate-pulse"></div>
              ))
            ) : processingQueue.length > 0 ? (
              processingQueue.map((item) => (
                <ProcessingItem 
                  key={item.id} 
                  item={item} 
                  dataset={item.dataset} 
                />
              ))
            ) : (
              <div className="text-center py-6 border border-neutral-light rounded-lg">
                <p className="text-neutral-medium">No datasets currently processing.</p>
              </div>
            )}
          </div>
          
          <div className="mt-4 text-center">
            <Link href="/add-dataset">
              <Button className="bg-primary text-white py-2 px-6 rounded-lg flex items-center justify-center mx-auto hover:bg-primary-dark">
                <Plus className="mr-2 h-5 w-5" />
                Add Dataset for Processing
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* FAIR Principles Section */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 md:p-6 border-b border-neutral-light">
          <h2 className="text-xl font-semibold mb-2">Metadata Quality Metrics</h2>
          <p className="text-neutral-medium text-sm">Overview of dataset compliance with FAIR principles and schema.org standards</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 md:p-6">
          {/* FAIR Compliance */}
          <ComplianceChart
            title="FAIR Compliance"
            metrics={fairMetrics}
            overallScore={85}
            color="#3949ab" // primary color
          />
          
          {/* Schema.org Compliance */}
          <ComplianceChart
            title="Schema.org Compliance"
            metrics={schemaOrgMetrics}
            overallScore={83}
            color="#00acc1" // secondary color
          />
        </div>
      </div>
    </div>
  );
}
