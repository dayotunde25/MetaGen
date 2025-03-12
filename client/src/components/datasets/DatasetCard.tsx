import { Dataset } from "@shared/schema";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getQualityColor, truncateText } from "@/lib/utils";
import { FileText, Database, Clock, Info, Download } from "lucide-react";

interface DatasetCardProps {
  dataset: Dataset;
  onDetailsClick: (dataset: Dataset) => void;
}

export default function DatasetCard({ dataset, onDetailsClick }: DatasetCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition">
      <div className="h-40 bg-slate-200 overflow-hidden relative">
        {dataset.thumbnail && (
          <img 
            src={dataset.thumbnail} 
            className="w-full h-full object-cover" 
            alt={`${dataset.name} thumbnail`}
          />
        )}
        <div className="absolute top-2 right-2 bg-white rounded-full px-2 py-1 text-xs font-semibold text-slate-700 shadow">
          {dataset.fairScore}
          <span className="ml-1 text-xs text-slate-500">FAIR Score</span>
        </div>
      </div>
      <div className="p-4">
        <div className="flex justify-between">
          <div>
            <h3 className="font-semibold text-slate-900">{dataset.name}</h3>
            <p className="text-sm text-slate-500 mt-1">{dataset.source}</p>
          </div>
          <div className="flex items-start">
            <Badge 
              className={getQualityColor(dataset.quality)}
            >
              {dataset.quality}
            </Badge>
          </div>
        </div>
        <p className="mt-2 text-sm text-slate-600 line-clamp-2">
          {truncateText(dataset.description, 120)}
        </p>
        <div className="mt-3 flex flex-wrap gap-1">
          {dataset.tags.map((tag, index) => (
            <Badge
              key={index}
              variant="outline"
              className="bg-slate-100 text-slate-800"
            >
              {tag}
            </Badge>
          ))}
        </div>
        <div className="mt-4 flex justify-between items-center text-sm text-slate-500">
          <div className="flex items-center">
            <FileText className="h-4 w-4 mr-1" />
            <span>{dataset.format}</span>
          </div>
          <div className="flex items-center">
            <Database className="h-4 w-4 mr-1" />
            <span>{dataset.size}</span>
          </div>
          <div className="flex items-center">
            <Clock className="h-4 w-4 mr-1" />
            <span>{dataset.lastUpdated}</span>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2">
          <Button 
            onClick={() => onDetailsClick(dataset)} 
            className="py-2 bg-primary-600 hover:bg-primary-700 text-white"
          >
            <Info className="h-4 w-4 mr-2" />
            <span>Details</span>
          </Button>
          <Button 
            variant="outline"
            className="py-2 border border-primary-600 text-primary-600 hover:bg-primary-50"
          >
            <Download className="h-4 w-4 mr-2" />
            <span>Download</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
