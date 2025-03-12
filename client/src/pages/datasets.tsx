import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import FilterSection from "@/components/dashboard/FilterSection";
import DatasetResults from "@/components/dashboard/DatasetResults";
import MetadataModal from "@/components/dashboard/MetadataModal";
import { DatabaseIcon, DownloadIcon, PlusIcon } from "lucide-react";

const Datasets = () => {
  const [filters, setFilters] = useState({
    dataType: "",
    category: "",
    sortBy: "relevance",
    tags: []
  });

  const handleFilterChange = (newFilters: any) => {
    setFilters(newFilters);
  };

  return (
    <main className="flex-grow">
      <div className="bg-primary-700 text-white">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-extrabold sm:text-3xl">
                Datasets
              </h1>
              <p className="mt-2 text-lg">
                Browse and search through our collection of research datasets
              </p>
            </div>
            <div className="mt-4 md:mt-0">
              <Button className="bg-white text-primary-700 hover:bg-gray-100">
                <PlusIcon className="mr-2 h-4 w-4" />
                Source New Dataset
              </Button>
            </div>
          </div>
        </div>
      </div>
      
      <FilterSection onFilterChange={handleFilterChange} />
      
      <DatasetResults />
    </main>
  );
};

export default Datasets;
