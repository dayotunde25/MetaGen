import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { 
  CheckCircleIcon 
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Dataset, Metadata } from "@shared/schema";

interface MetadataModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset: Dataset;
  metadata: Metadata | null;
  isLoading: boolean;
}

const MetadataModal: React.FC<MetadataModalProps> = ({ isOpen, onClose, dataset, metadata, isLoading }) => {
  if (!isOpen) return null;
  
  const fairScores = metadata?.fairScores as Record<string, number> | undefined;

  const downloadMetadata = () => {
    if (!metadata) return;
    
    const dataStr = JSON.stringify(metadata.schemaOrgJson, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `${dataset.title.replace(/\s+/g, '_')}_metadata.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Dataset Metadata: {dataset.title}</DialogTitle>
        </DialogHeader>
        
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        ) : metadata ? (
          <div className="mt-4">
            <div className="bg-gray-50 p-4 rounded-md mb-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-gray-900">FAIR Compliance Assessment</h3>
                  <div className="mt-2 text-sm text-gray-500 flex flex-wrap items-center gap-4">
                    {fairScores && Object.entries(fairScores).map(([key, value]) => (
                      <div key={key} className="flex mr-4">
                        <span className="font-medium text-green-600 mr-1">{key.charAt(0).toUpperCase()}</span>
                        <div className="w-24 bg-gray-200 rounded-full h-2 my-auto">
                          <div 
                            className="bg-green-500 h-2 rounded-full" 
                            style={{ width: `${value}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            <ul className="divide-y divide-gray-200">
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Title:</span>
                  <span className="text-sm text-gray-900">{dataset.title}</span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Description:</span>
                  <span className="text-sm text-gray-900">{dataset.description}</span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Creator:</span>
                  <span className="text-sm text-gray-900">{dataset.source}</span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Publication Date:</span>
                  <span className="text-sm text-gray-900">
                    {new Date(dataset.updatedAt).toISOString().split('T')[0]}
                  </span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">License:</span>
                  <span className="text-sm text-gray-900">{metadata.license || "Unknown"}</span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Format:</span>
                  <span className="text-sm text-gray-900">{dataset.format}</span>
                </div>
              </li>
              <li className="py-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium text-gray-500">Size:</span>
                  <span className="text-sm text-gray-900">
                    {dataset.size}
                    {dataset.recordCount ? ` (${dataset.recordCount.toLocaleString()} records)` : ''}
                  </span>
                </div>
              </li>
            </ul>
            
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-900 mb-2">Structured Metadata (schema.org)</h4>
              <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto max-h-60 font-mono text-xs">
                <pre><code>{JSON.stringify(metadata.schemaOrgJson, null, 2)}</code></pre>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center">
            <p className="text-gray-500">No metadata available for this dataset.</p>
          </div>
        )}

        <DialogFooter>
          <Button 
            variant="outline" 
            onClick={onClose}
          >
            Close
          </Button>
          {metadata && (
            <Button 
              variant="default"
              onClick={downloadMetadata}
            >
              Download Metadata
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default MetadataModal;
