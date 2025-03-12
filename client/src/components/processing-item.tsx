import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ProcessingQueue, Dataset } from "@shared/schema";
import ProgressBar from "./progress-bar";

interface ProcessingItemProps {
  item: ProcessingQueue;
  dataset: Dataset;
}

export default function ProcessingItem({ item, dataset }: ProcessingItemProps) {
  // Format the estimated time remaining
  const formatEstimatedTime = () => {
    if (!item.estimatedCompletionTime) {
      return "Unknown";
    }
    
    const now = new Date();
    const estimatedCompletion = new Date(item.estimatedCompletionTime);
    
    if (estimatedCompletion <= now) {
      return "Almost done";
    }
    
    const diffMs = estimatedCompletion.getTime() - now.getTime();
    const diffMins = Math.round(diffMs / 60000);
    
    if (diffMins < 1) {
      return "Less than a minute";
    } else if (diffMins === 1) {
      return "1 minute";
    } else if (diffMins < 60) {
      return `${diffMins} minutes`;
    } else {
      const hours = Math.floor(diffMins / 60);
      const mins = diffMins % 60;
      return `${hours} hour${hours !== 1 ? 's' : ''}${mins > 0 ? ` ${mins} minute${mins !== 1 ? 's' : ''}` : ''}`;
    }
  };

  // Get badge color based on status
  const getStatusBadge = () => {
    switch (item.status) {
      case "processing":
        return <Badge className="bg-status-info">Processing</Badge>;
      case "completed":
        return <Badge className="bg-status-success">Completed</Badge>;
      case "error":
        return <Badge className="bg-destructive">Error</Badge>;
      case "queued":
      default:
        return <Badge className="bg-neutral-medium">Queued</Badge>;
    }
  };

  return (
    <Card className="border border-neutral-light">
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-medium">{dataset.name}</h3>
          {getStatusBadge()}
        </div>
        
        <div className="flex items-center text-sm text-neutral-medium mb-3">
          {dataset.size && (
            <span className="mr-4 flex items-center">
              <span className="material-icons text-sm mr-1">memory</span>
              {dataset.size}
            </span>
          )}
          
          {dataset.format && (
            <span className="flex items-center">
              <span className="material-icons text-sm mr-1">description</span>
              {dataset.format}
            </span>
          )}
        </div>
        
        <div className="relative pt-1">
          <div className="flex mb-1 items-center justify-between">
            <div>
              <span className="text-xs font-medium text-neutral-dark">
                {item.status === "queued" ? "In queue" : item.status === "processing" ? "Downloading" : item.status}
              </span>
            </div>
            <div className="text-right">
              {item.status === "queued" ? (
                <span className="text-xs font-medium text-neutral-dark">
                  Position: {/* This would come from the queue position in a real app */}
                </span>
              ) : (
                <span className="text-xs font-medium text-neutral-dark">
                  {item.progress}%
                </span>
              )}
            </div>
          </div>
          
          <ProgressBar 
            value={item.progress || 0} 
            status={item.status} 
          />
          
          <div className="text-xs text-neutral-medium mt-2">
            {item.status === "queued" ? (
              <>Estimated start: {formatEstimatedTime()}</>
            ) : item.status === "processing" ? (
              <>Estimated time remaining: {formatEstimatedTime()}</>
            ) : item.status === "completed" ? (
              <>Completed {item.endTime ? new Date(item.endTime).toLocaleString() : ''}</>
            ) : item.status === "error" ? (
              <>Error: {item.error}</>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
