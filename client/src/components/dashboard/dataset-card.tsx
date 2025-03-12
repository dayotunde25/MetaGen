import { useState } from "react";
import { FileText, Download, MoreHorizontal } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import ProgressBar from "@/components/ui/progress-bar";
import MetadataPreviewModal from "@/components/metadata/metadata-preview-modal";

interface Dataset {
  id: number;
  title: string;
  description?: string;
  source: string;
  category?: string;
  size?: string;
  formats?: string[];
  status: string;
  progress?: number;
  estimatedTimeToCompletion?: string;
  updatedAt: Date;
}

interface DatasetCardProps {
  dataset: Dataset;
}

export default function DatasetCard({ dataset }: DatasetCardProps) {
  const [showMetadataModal, setShowMetadataModal] = useState(false);
  
  const getStatusBadge = () => {
    switch (dataset.status) {
      case 'processed':
        return (
          <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Processed
          </Badge>
        );
      case 'downloading':
        return (
          <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">
            <svg className="w-3 h-3 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Downloading
          </Badge>
        );
      case 'structuring':
        return (
          <Badge className="bg-purple-100 text-purple-800 hover:bg-purple-100">
            <svg className="w-3 h-3 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Structuring
          </Badge>
        );
      case 'generating':
        return (
          <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">
            <svg className="w-3 h-3 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Generating Metadata
          </Badge>
        );
      case 'failed':
        return (
          <Badge className="bg-red-100 text-red-800 hover:bg-red-100">
            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Failed
          </Badge>
        );
      default:
        return (
          <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-100">
            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Pending
          </Badge>
        );
    }
  };
  
  const getProgressColor = () => {
    switch (dataset.status) {
      case 'downloading':
        return 'amber';
      case 'structuring':
        return 'purple';
      case 'generating':
        return 'blue';
      default:
        return 'primary';
    }
  };
  
  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - new Date(date).getTime();
    
    // Less than a day
    if (diff < 86400000) {
      const hours = Math.floor(diff / 3600000);
      if (hours < 1) return 'Just now';
      return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
    }
    
    // Less than a week
    if (diff < 604800000) {
      const days = Math.floor(diff / 86400000);
      return `${days} ${days === 1 ? 'day' : 'days'} ago`;
    }
    
    // Default to date
    return new Date(date).toLocaleDateString();
  };
  
  return (
    <>
      <Card className="bg-white shadow rounded-lg overflow-hidden">
        <CardContent className="p-0">
          <div className="p-5 border-b border-gray-200">
            <div className="flex justify-between items-start">
              <div>
                <Badge variant="outline" className="mb-2">
                  {dataset.category || 'Uncategorized'}
                </Badge>
                <h3 className="text-base font-semibold text-gray-900 leading-tight">
                  {dataset.title}
                </h3>
                <p className="text-sm text-gray-500 mt-1">{dataset.source}</p>
              </div>
              {getStatusBadge()}
            </div>
            
            <div className="mt-3 text-sm text-gray-600">
              <p className="line-clamp-2">{dataset.description}</p>
            </div>
          </div>
          
          <div className="px-5 py-3 bg-gray-50 text-xs">
            <div className="flex flex-wrap gap-2">
              {dataset.formats?.map((format, index) => (
                <Badge key={index} variant="outline" className="bg-blue-50 text-blue-700 hover:bg-blue-50">
                  {format}
                </Badge>
              ))}
              
              {dataset.status === 'processed' && (
                <Badge variant="outline" className="bg-purple-50 text-purple-700 hover:bg-purple-50">
                  FAIR Compliant
                </Badge>
              )}
              
              {dataset.size && (
                <Badge variant="outline" className="bg-gray-100 text-gray-800 hover:bg-gray-100">
                  <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  {dataset.size}
                </Badge>
              )}
            </div>
          </div>
          
          {['downloading', 'structuring', 'generating'].includes(dataset.status) && dataset.progress !== undefined && (
            <div className="px-5 py-3 border-t border-gray-200 flex justify-between items-center">
              <span className="text-xs text-gray-500">
                {dataset.status === 'downloading' && 'Downloading...'}
                {dataset.status === 'structuring' && 'Structuring...'}
                {dataset.status === 'generating' && 'Generating Metadata...'}
                {' '}({dataset.progress}%)
              </span>
              <div className="w-24">
                <ProgressBar value={dataset.progress} color={getProgressColor()} height="sm" />
              </div>
            </div>
          )}
          
          {!['downloading', 'structuring', 'generating'].includes(dataset.status) && (
            <div className="px-5 py-3 border-t border-gray-200 flex justify-between items-center">
              <span className="text-xs text-gray-500">
                Updated {formatDate(dataset.updatedAt)}
              </span>
              <div className="flex space-x-2">
                {dataset.status === 'processed' && (
                  <button
                    onClick={() => setShowMetadataModal(true)}
                    className="text-sm font-medium text-primary-600 hover:text-primary-700"
                  >
                    <FileText className="h-4 w-4" />
                  </button>
                )}
                <button
                  disabled={dataset.status !== 'processed'}
                  className={`text-sm font-medium ${
                    dataset.status === 'processed'
                      ? 'text-primary-600 hover:text-primary-700'
                      : 'text-gray-400 cursor-not-allowed'
                  }`}
                >
                  <Download className="h-4 w-4" />
                </button>
                <button className="text-sm font-medium text-primary-600 hover:text-primary-700">
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {showMetadataModal && (
        <MetadataPreviewModal
          datasetId={dataset.id}
          isOpen={showMetadataModal}
          onClose={() => setShowMetadataModal(false)}
        />
      )}
    </>
  );
}
