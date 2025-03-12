import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dataset } from "@shared/schema";
import { formatDistanceToNow } from "date-fns";
import { Link } from "wouter";

interface DatasetCardProps {
  dataset: Dataset;
}

export default function DatasetCard({ dataset }: DatasetCardProps) {
  // Helper to determine quality badge color
  const getQualityBadge = (score?: number) => {
    if (!score) return null;
    
    if (score >= 80) {
      return <Badge className="bg-status-success">High Quality</Badge>;
    } else if (score >= 60) {
      return <Badge className="bg-status-info">Medium Quality</Badge>;
    } else {
      return <Badge className="bg-status-warning">Low Quality</Badge>;
    }
  };

  // Format the date added to a relative time
  const formatDate = (dateString: string | Date) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : dateString;
      return formatDistanceToNow(date, { addSuffix: true });
    } catch (error) {
      return 'Unknown date';
    }
  };

  return (
    <Card className="overflow-hidden transition-all hover:shadow-md hover:translate-y-[-2px]">
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-3">
          <h3 className="font-semibold text-lg truncate" title={dataset.name}>
            {dataset.name}
          </h3>
          {getQualityBadge(dataset.qualityScore)}
        </div>
        
        <p className="text-sm text-neutral-dark mb-4 line-clamp-2" title={dataset.description}>
          {dataset.description || "No description available."}
        </p>
        
        <div className="flex flex-wrap mb-4">
          {dataset.format && (
            <span className="text-xs bg-neutral-lightest px-2 py-1 rounded-md mr-2 mb-2">
              {dataset.format}
            </span>
          )}
          
          {dataset.tags && dataset.tags.map((tag, index) => (
            <span key={index} className="text-xs bg-neutral-lightest px-2 py-1 rounded-md mr-2 mb-2">
              {tag}
            </span>
          ))}
        </div>
        
        <div className="flex justify-between items-center text-sm text-neutral-medium">
          <span className="flex items-center">
            <span className="material-icons text-sm mr-1">source</span>
            {dataset.source || "Unknown Source"}
          </span>
          <span className="flex items-center">
            <span className="material-icons text-sm mr-1">calendar_today</span>
            Added {formatDate(dataset.dateAdded)}
          </span>
        </div>
      </CardContent>
      
      <CardFooter className="bg-neutral-lightest p-4 flex justify-between">
        <div>
          {dataset.isFairCompliant && (
            <span className="inline-flex items-center text-xs mr-3">
              <span className="material-icons text-status-success text-sm mr-1">check_circle</span>
              FAIR Compliant
            </span>
          )}
          
          {dataset.schemaOrgScore && dataset.schemaOrgScore >= 70 ? (
            <span className="inline-flex items-center text-xs">
              <span className="material-icons text-status-success text-sm mr-1">check_circle</span>
              schema.org
            </span>
          ) : dataset.schemaOrgScore && dataset.schemaOrgScore >= 40 ? (
            <span className="inline-flex items-center text-xs">
              <span className="material-icons text-status-warning text-sm mr-1">warning</span>
              schema.org partial
            </span>
          ) : null}
        </div>
        
        <Link href={`/dataset/${dataset.id}`}>
          <Button variant="ghost" size="icon" className="text-primary hover:text-primary-dark">
            <Eye className="h-5 w-5" />
          </Button>
        </Link>
      </CardFooter>
    </Card>
  );
}
