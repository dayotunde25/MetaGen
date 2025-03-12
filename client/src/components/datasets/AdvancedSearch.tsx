import { X } from "lucide-react";
import { Button } from "../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { useState } from "react";

interface AdvancedSearchProps {
  onClose: () => void;
}

export default function AdvancedSearch({ onClose }: AdvancedSearchProps) {
  const [filters, setFilters] = useState({
    datasetType: "",
    fairScore: "",
    lastUpdated: "",
    licenseType: "",
    dataFormat: "",
    datasetSize: "",
  });

  const handleFilterChange = (key: string, value: string) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const resetFilters = () => {
    setFilters({
      datasetType: "",
      fairScore: "",
      lastUpdated: "",
      licenseType: "",
      dataFormat: "",
      datasetSize: "",
    });
  };

  const applyFilters = () => {
    // Here you would apply the filters to your search
    console.log("Applying filters:", filters);
    onClose();
  };

  return (
    <div className="bg-white rounded-lg shadow mb-6 p-4" x-transition>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium">Advanced Search</h3>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-500">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Dataset Type</label>
          <Select value={filters.datasetType} onValueChange={(value) => handleFilterChange("datasetType", value)}>
            <SelectTrigger>
              <SelectValue placeholder="All Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All Types</SelectItem>
              <SelectItem value="tabular">Tabular Data</SelectItem>
              <SelectItem value="image">Image Collections</SelectItem>
              <SelectItem value="text">Text Corpora</SelectItem>
              <SelectItem value="time-series">Time Series</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">FAIR Score</label>
          <Select value={filters.fairScore} onValueChange={(value) => handleFilterChange("fairScore", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Any Score" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Any Score</SelectItem>
              <SelectItem value="90+">90+ (Excellent)</SelectItem>
              <SelectItem value="80-89">80-89 (Good)</SelectItem>
              <SelectItem value="70-79">70-79 (Adequate)</SelectItem>
              <SelectItem value="<70">Below 70 (Needs Improvement)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Last Updated</label>
          <Select value={filters.lastUpdated} onValueChange={(value) => handleFilterChange("lastUpdated", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Any Time" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Any Time</SelectItem>
              <SelectItem value="week">Past Week</SelectItem>
              <SelectItem value="month">Past Month</SelectItem>
              <SelectItem value="quarter">Past 3 Months</SelectItem>
              <SelectItem value="year">Past Year</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">License Type</label>
          <Select value={filters.licenseType} onValueChange={(value) => handleFilterChange("licenseType", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Any License" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Any License</SelectItem>
              <SelectItem value="open">Open (CC0, Public Domain)</SelectItem>
              <SelectItem value="attribution">Attribution Required (CC-BY)</SelectItem>
              <SelectItem value="non-commercial">Non-Commercial</SelectItem>
              <SelectItem value="research">Research Use Only</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Data Format</label>
          <Select value={filters.dataFormat} onValueChange={(value) => handleFilterChange("dataFormat", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Any Format" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Any Format</SelectItem>
              <SelectItem value="csv">CSV</SelectItem>
              <SelectItem value="json">JSON</SelectItem>
              <SelectItem value="excel">Excel</SelectItem>
              <SelectItem value="xml">XML</SelectItem>
              <SelectItem value="parquet">Parquet</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Dataset Size</label>
          <Select value={filters.datasetSize} onValueChange={(value) => handleFilterChange("datasetSize", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Any Size" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Any Size</SelectItem>
              <SelectItem value="small">Small (&lt;100MB)</SelectItem>
              <SelectItem value="medium">Medium (100MB-1GB)</SelectItem>
              <SelectItem value="large">Large (1GB-10GB)</SelectItem>
              <SelectItem value="very-large">Very Large (&gt;10GB)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="mt-4 flex justify-end space-x-3">
        <Button variant="outline" onClick={resetFilters}>Reset</Button>
        <Button onClick={applyFilters}>Apply Filters</Button>
      </div>
    </div>
  );
}
