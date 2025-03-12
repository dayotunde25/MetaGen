import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Plus, Loader2, FileDown, FileUp, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DataTable } from "@/components/ui/data-table";
import { formatDistanceToNow } from "date-fns";
import { Badge } from "@/components/ui/badge";
import { Dataset } from "@shared/schema";

export default function MyDatasets() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);

  // Fetch user's datasets
  const { data: datasets, isLoading } = useQuery<Dataset[]>({
    queryKey: ["/api/datasets"],
  });

  // Filter datasets based on search query and status filter
  const filteredDatasets = datasets?.filter((dataset) => {
    const matchesSearch = searchQuery 
      ? dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (dataset.description?.toLowerCase().includes(searchQuery.toLowerCase()))
      : true;
    
    const matchesStatus = statusFilter
      ? dataset.status === statusFilter
      : true;
    
    return matchesSearch && matchesStatus;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "processed":
        return <Badge className="bg-status-success">Processed</Badge>;
      case "processing":
        return <Badge className="bg-status-info">Processing</Badge>;
      case "error":
        return <Badge className="bg-destructive">Error</Badge>;
      case "queued":
        return <Badge className="bg-neutral-medium">Queued</Badge>;
      default:
        return <Badge className="bg-neutral-light">{status}</Badge>;
    }
  };

  const columns = [
    { 
      header: "Name", 
      accessorKey: "name" as keyof Dataset,
      cell: (dataset: Dataset) => (
        <Link href={`/dataset/${dataset.id}`}>
          <a className="text-primary hover:underline font-medium">{dataset.name}</a>
        </Link>
      )
    },
    { 
      header: "Source", 
      accessorKey: "source" as keyof Dataset,
      cell: (dataset: Dataset) => dataset.source || "Unknown"
    },
    { 
      header: "Format", 
      accessorKey: "format" as keyof Dataset,
      cell: (dataset: Dataset) => dataset.format || "Unknown"
    },
    { 
      header: "Status", 
      accessorKey: "status" as keyof Dataset,
      cell: (dataset: Dataset) => getStatusBadge(dataset.status || "Unknown")
    },
    { 
      header: "Quality", 
      accessorKey: "qualityScore" as keyof Dataset,
      cell: (dataset: Dataset) => {
        if (!dataset.qualityScore) return "Not evaluated";
        if (dataset.qualityScore >= 80) return "High";
        if (dataset.qualityScore >= 60) return "Medium";
        return "Low";
      }
    },
    { 
      header: "Date Added", 
      accessorKey: "dateAdded" as keyof Dataset,
      cell: (dataset: Dataset) => {
        const date = dataset.dateAdded instanceof Date 
          ? dataset.dateAdded 
          : new Date(dataset.dateAdded);
        return formatDistanceToNow(date, { addSuffix: true });
      }
    },
    { 
      header: "Actions", 
      accessorKey: "id" as keyof Dataset,
      cell: (dataset: Dataset) => (
        <div className="flex space-x-2">
          <Button variant="ghost" size="sm" className="p-1 h-8 w-8">
            <FileDown className="h-4 w-4" />
          </Button>
          
          <Button variant="ghost" size="sm" className="p-1 h-8 w-8">
            <FileUp className="h-4 w-4" />
          </Button>
        </div>
      )
    },
  ];

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-semibold">My Datasets</h1>
        
        <Link href="/add-dataset">
          <Button className="bg-primary hover:bg-primary-dark text-white">
            <Plus className="mr-2 h-4 w-4" /> Add Dataset
          </Button>
        </Link>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="relative flex-grow">
            <Input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          
          <div className="flex space-x-2">
            <Button
              variant={statusFilter === null ? "default" : "outline"}
              onClick={() => setStatusFilter(null)}
              className="min-w-[100px]"
            >
              All
            </Button>
            <Button
              variant={statusFilter === "processed" ? "default" : "outline"}
              onClick={() => setStatusFilter("processed")}
              className="min-w-[100px]"
            >
              Processed
            </Button>
            <Button
              variant={statusFilter === "processing" ? "default" : "outline"}
              onClick={() => setStatusFilter("processing")}
              className="min-w-[100px]"
            >
              Processing
            </Button>
            <Button
              variant={statusFilter === "queued" ? "default" : "outline"}
              onClick={() => setStatusFilter("queued")}
              className="min-w-[100px]"
            >
              Queued
            </Button>
          </div>
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={filteredDatasets || []}
          />
        )}
      </div>
    </div>
  );
}
