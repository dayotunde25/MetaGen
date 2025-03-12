import { useState } from "react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  DatabaseIcon, 
  EyeIcon, 
  CodeIcon, 
  DownloadIcon, 
  ClockIcon 
} from "lucide-react";
import { Dataset, Metadata } from "@shared/schema";
import MetadataModal from "./MetadataModal";

interface DatasetCardProps {
  dataset: Dataset;
  onViewMetadata: (dataset: Dataset) => void;
}

const DatasetCard: React.FC<DatasetCardProps> = ({ dataset, onViewMetadata }) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const { toast } = useToast();

  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      const res = await apiRequest("GET", `/api/datasets/${dataset.id}/download`, undefined);
      const data = await res.json();
      
      if (data.success) {
        toast({
          title: "Download initiated",
          description: "Your dataset download has started",
        });
        
        // In a real app, we would trigger the actual download here
        window.open(data.downloadUrl, "_blank");
      }
    } catch (error) {
      toast({
        title: "Download failed",
        description: "There was an error downloading the dataset",
        variant: "destructive"
      });
    } finally {
      setIsDownloading(false);
    }
  };

  const formatDate = (dateString: string | Date) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  return (
    <Card className="bg-white shadow overflow-hidden sm:rounded-lg mb-6">
      <div className="px-4 py-5 sm:px-6 flex justify-between">
        <div>
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            {dataset.title}
          </h3>
          <p className="max-w-2xl text-sm text-gray-500 mt-1">
            {dataset.description}
          </p>
        </div>
        <div className="flex gap-2">
          {dataset.fairCompliant && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium bg-green-100 text-green-800">
              FAIR
            </span>
          )}
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium bg-blue-100 text-blue-800">
            {dataset.dataType}
          </span>
        </div>
      </div>
      <div className="border-t border-gray-200 px-4 py-5 sm:p-0">
        <dl className="sm:divide-y sm:divide-gray-200">
          <div className="sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
            <dt className="text-sm font-medium text-gray-500">Source</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              <div className="flex items-center">
                <DatabaseIcon className="mr-2 h-4 w-4 text-gray-400" />
                <span>{dataset.source}</span>
              </div>
            </dd>
          </div>
          <div className="sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
            <dt className="text-sm font-medium text-gray-500">Metadata Quality</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full ${dataset.metadataQuality >= 80 ? 'bg-green-500' : dataset.metadataQuality >= 60 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                    style={{ width: `${dataset.metadataQuality}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-sm font-medium text-gray-700">{dataset.metadataQuality}%</span>
              </div>
            </dd>
          </div>
          <div className="sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
            <dt className="text-sm font-medium text-gray-500">Format</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              <div className="flex items-center">
                {dataset.format?.split(', ').map((format, index) => (
                  <span key={index} className={`${index > 0 ? 'ml-2' : ''} px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800`}>
                    {format}
                  </span>
                ))}
              </div>
            </dd>
          </div>
          <div className="sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
            <dt className="text-sm font-medium text-gray-500">Size</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              <span>
                {dataset.size} 
                {dataset.recordCount ? ` (${dataset.recordCount.toLocaleString()} records)` : ''}
              </span>
            </dd>
          </div>
        </dl>
      </div>
      <div className="bg-gray-50 px-4 py-4 sm:px-6 flex justify-between items-center">
        <div className="flex items-center text-sm text-gray-500">
          <ClockIcon className="mr-1 h-4 w-4" />
          <span>Last updated: {formatDate(dataset.updatedAt)}</span>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" size="sm" className="inline-flex items-center">
            <EyeIcon className="mr-2 h-4 w-4" />
            Preview
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            className="inline-flex items-center"
            onClick={() => onViewMetadata(dataset)}
          >
            <CodeIcon className="mr-2 h-4 w-4" />
            Metadata
          </Button>
          <Button 
            variant="default" 
            size="sm" 
            className="inline-flex items-center"
            onClick={handleDownload}
            disabled={isDownloading}
          >
            <DownloadIcon className="mr-2 h-4 w-4" />
            {isDownloading ? "Downloading..." : "Download"}
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default DatasetCard;
