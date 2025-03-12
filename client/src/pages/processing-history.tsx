import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { 
  Download, 
  ListChecks, 
  FileSpreadsheet,
  CheckCircle, 
  XCircle, 
  Clock 
} from "lucide-react";

export default function ProcessingHistory() {
  const { toast } = useToast();
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  
  const { data: allDatasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ['/api/datasets'],
  });
  
  const { data: historyEntries, isLoading: historyLoading } = useQuery({
    queryKey: ['/api/datasets', selectedDataset, 'history'],
    queryFn: async () => {
      if (!selectedDataset) return [];
      const res = await fetch(`/api/datasets/${selectedDataset}/history`);
      if (!res.ok) throw new Error('Failed to fetch history');
      return await res.json();
    },
    enabled: !!selectedDataset,
  });
  
  const getOperationIcon = (operation: string) => {
    switch (operation) {
      case 'download':
        return <Download className="h-4 w-4 text-blue-500" />;
      case 'structure':
        return <ListChecks className="h-4 w-4 text-purple-500" />;
      case 'metadata_generation':
        return <FileSpreadsheet className="h-4 w-4 text-green-500" />;
      default:
        return null;
    }
  };
  
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return (
          <Badge variant="outline" className="bg-green-50 text-green-700 hover:bg-green-50">
            <CheckCircle className="h-3 w-3 mr-1" />
            Success
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="outline" className="bg-red-50 text-red-700 hover:bg-red-50">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      case 'in_progress':
        return (
          <Badge variant="outline" className="bg-amber-50 text-amber-700 hover:bg-amber-50">
            <Clock className="h-3 w-3 mr-1" />
            In Progress
          </Badge>
        );
      default:
        return null;
    }
  };
  
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };
  
  const calculateDuration = (startTime: string, endTime?: string) => {
    if (!endTime) return "In progress";
    
    const start = new Date(startTime).getTime();
    const end = new Date(endTime).getTime();
    const durationMs = end - start;
    
    // Format the duration
    if (durationMs < 60000) {
      return `${Math.round(durationMs / 1000)}s`;
    } else if (durationMs < 3600000) {
      return `${Math.round(durationMs / 60000)}m`;
    } else {
      return `${Math.round(durationMs / 3600000)}h ${Math.round((durationMs % 3600000) / 60000)}m`;
    }
  };
  
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-gray-800">Processing History</h1>
      
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Select Dataset</CardTitle>
        </CardHeader>
        <CardContent>
          <Select value={selectedDataset} onValueChange={setSelectedDataset}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose a dataset to view its processing history" />
            </SelectTrigger>
            <SelectContent>
              {!datasetsLoading && allDatasets?.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id.toString()}>
                  {dataset.title}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Processing Operations</CardTitle>
        </CardHeader>
        <CardContent>
          {!selectedDataset ? (
            <div className="text-center py-6">
              <p className="text-gray-500">Select a dataset to view its processing history</p>
            </div>
          ) : historyLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="h-10 bg-gray-200 rounded"></div>
              <div className="h-10 bg-gray-200 rounded"></div>
              <div className="h-10 bg-gray-200 rounded"></div>
            </div>
          ) : historyEntries?.length === 0 ? (
            <div className="text-center py-6">
              <p className="text-gray-500">No processing history available for this dataset</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Operation</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Start Time</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {historyEntries?.map((entry) => (
                  <TableRow key={entry.id}>
                    <TableCell className="flex items-center space-x-2">
                      {getOperationIcon(entry.operation)}
                      <span className="capitalize">
                        {entry.operation.replace('_', ' ')}
                      </span>
                    </TableCell>
                    <TableCell>{getStatusBadge(entry.status)}</TableCell>
                    <TableCell>{formatDate(entry.startTime)}</TableCell>
                    <TableCell>{calculateDuration(entry.startTime, entry.endTime)}</TableCell>
                    <TableCell className="max-w-xs truncate">{entry.details || "—"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
