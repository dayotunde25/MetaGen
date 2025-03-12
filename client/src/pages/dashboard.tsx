import { useState } from "react";
import Hero from "@/components/dashboard/Hero";
import Stats from "@/components/dashboard/Stats";
import FilterSection from "@/components/dashboard/FilterSection";
import DatasetResults from "@/components/dashboard/DatasetResults";
import Features from "@/components/dashboard/Features";

const Dashboard = () => {
  const [filters, setFilters] = useState({
    dataType: "",
    category: "",
    sortBy: "relevance",
    tags: ["FAIR Compliant"]
  });

  const handleFilterChange = (newFilters: any) => {
    setFilters(newFilters);
  };

  return (
    <main className="flex-grow">
      <Hero />
      <Stats />
      <FilterSection onFilterChange={handleFilterChange} />
      <DatasetResults />
      <Features />
    </main>
  );
};

export default Dashboard;
