import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchDatasets, fetchCategories, searchDatasets } from "@/lib/api";
import { Dataset } from "@shared/schema";
import DatasetCard from "@/components/datasets/DatasetCard";
import DatasetDetails from "@/components/datasets/DatasetDetails";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ChevronDown, BarChart, Leaf, Heart, GraduationCap } from "lucide-react";

export default function Discover() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: datasets = [], isLoading } = useQuery({
    queryKey: ['/api/datasets', searchQuery],
    queryFn: () => searchQuery ? searchDatasets(searchQuery) : fetchDatasets()
  });

  const { data: categories = [] } = useQuery({
    queryKey: ['/api/categories'],
    queryFn: fetchCategories,
    initialData: [
      { name: "Economics", count: 215, icon: "BarChart", color: "text-secondary-500" },
      { name: "Environment", count: 189, icon: "Leaf", color: "text-green-500" },
      { name: "Healthcare", count: 312, icon: "Heart", color: "text-red-500" },
      { name: "Education", count: 156, icon: "GraduationCap", color: "text-amber-500" }
    ]
  });

  const getIconComponent = (iconName: string) => {
    switch (iconName) {
      case "BarChart":
        return <BarChart className="text-2xl" />;
      case "Leaf":
        return <Leaf className="text-2xl" />;
      case "Heart":
        return <Heart className="text-2xl" />;
      case "GraduationCap":
        return <GraduationCap className="text-2xl" />;
      default:
        return <BarChart className="text-2xl" />;
    }
  };

  return (
    <div>
      {!selectedDataset ? (
        <>
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold text-slate-900">Discover Research Datasets</h1>
            <div className="flex space-x-2">
              <Button variant="outline" className="flex items-center">
                <BarChart className="mr-2 h-4 w-4" />
                <span>Sort</span>
              </Button>
              <Button variant="outline" className="flex items-center sm:hidden">
                <Search className="mr-2 h-4 w-4" />
                <span>Search</span>
              </Button>
            </div>
          </div>

          {/* Popular categories */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-4 text-slate-800">Popular Categories</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {categories.map((category, index) => (
                <button 
                  key={index}
                  className="bg-white rounded-lg shadow-sm p-4 border border-slate-200 hover:border-primary-400 transition"
                >
                  <div className={`${category.color} mb-2`}>
                    {getIconComponent(category.icon)}
                  </div>
                  <h3 className="font-medium">{category.name}</h3>
                  <p className="text-sm text-slate-500 mt-1">{category.count} datasets</p>
                </button>
              ))}
            </div>
          </div>

          {/* Dataset cards */}
          <div>
            <h2 className="text-lg font-semibold mb-4 text-slate-800">Recommended for You</h2>
            {isLoading ? (
              <p>Loading datasets...</p>
            ) : datasets.length === 0 ? (
              <Card className="p-8 text-center">
                <p className="text-slate-500">No datasets found. Try adjusting your search criteria.</p>
              </Card>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {datasets.map((dataset) => (
                    <DatasetCard 
                      key={dataset.id} 
                      dataset={dataset} 
                      onDetailsClick={setSelectedDataset} 
                    />
                  ))}
                </div>
                <div className="mt-6 flex justify-center">
                  <Button 
                    variant="link" 
                    className="flex items-center space-x-2 text-primary-600 font-medium hover:text-primary-700"
                  >
                    <span>Load more datasets</span>
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </div>
              </>
            )}
          </div>
        </>
      ) : (
        <DatasetDetails dataset={selectedDataset} onClose={() => setSelectedDataset(null)} />
      )}
    </div>
  );
}
