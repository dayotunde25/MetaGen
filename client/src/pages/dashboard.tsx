import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import StatCard from "@/components/dashboard/stat-card";
import SearchSection from "@/components/dashboard/search-section";
import DatasetCard from "@/components/dashboard/dataset-card";
import ProcessingQueue from "@/components/dashboard/processing-queue";
import { ArrowRight } from "lucide-react";
import { Link } from "wouter";

export default function Dashboard() {
  const { toast } = useToast();
  
  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['/api/statistics'],
  });
  
  const { data: recentDatasets, isLoading: datasetsLoading, error: datasetsError } = useQuery({
    queryKey: ['/api/datasets/recent'],
  });
  
  const { data: processingDatasets, isLoading: processingLoading, error: processingError } = useQuery({
    queryKey: ['/api/datasets/status/processing'],
  });
  
  if (statsError || datasetsError || processingError) {
    toast({
      title: "Error",
      description: "Failed to load dashboard data. Please try again.",
      variant: "destructive",
    });
  }
  
  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="Total Datasets"
          value={statsLoading ? "Loading..." : stats?.total.toString() || "0"}
          icon="database"
          change={stats?.totalChange || 0}
          loading={statsLoading}
        />
        
        <StatCard 
          title="Processed Datasets"
          value={statsLoading ? "Loading..." : stats?.processed.toString() || "0"}
          icon="check"
          change={stats?.processedChange || 0}
          color="green"
          loading={statsLoading}
        />
        
        <StatCard 
          title="Ongoing Processing"
          value={statsLoading ? "Loading..." : stats?.processing.toString() || "0"}
          icon="loader"
          timeEstimate="Est. completion: 2h"
          color="amber"
          loading={statsLoading}
        />
        
        <StatCard 
          title="Failed Operations"
          value={statsLoading ? "Loading..." : stats?.failed.toString() || "0"}
          icon="alert-triangle"
          change={stats?.failedChange || 0}
          color="red"
          loading={statsLoading}
        />
      </div>

      {/* Search Section */}
      <SearchSection />

      {/* Recent Datasets */}
      <div>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Recent Datasets</h2>
          <Link href="/downloaded">
            <a className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center">
              View All <ArrowRight className="ml-1 h-4 w-4" />
            </a>
          </Link>
        </div>
        
        {datasetsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
            <div className="h-56 bg-white shadow rounded-lg animate-pulse"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recentDatasets?.map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
            
            {(!recentDatasets || recentDatasets.length === 0) && (
              <div className="col-span-3 bg-white shadow rounded-lg p-6 text-center">
                <p className="text-gray-500">No datasets available yet. Start by searching for datasets.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Processing Queue */}
      <ProcessingQueue 
        datasets={processingDatasets || []} 
        isLoading={processingLoading} 
      />
    </div>
  );
}
