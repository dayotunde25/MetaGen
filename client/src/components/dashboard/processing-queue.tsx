import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import ProgressBar from "@/components/ui/progress-bar";
import { Badge } from "@/components/ui/badge";
import { Pause, PlusCircle, X } from "lucide-react";

interface Dataset {
  id: number;
  title: string;
  description?: string;
  source: string;
  sourceUrl: string;
  size?: string;
  status: string;
  progress?: number;
  estimatedTimeToCompletion?: string;
}

interface ProcessingQueueProps {
  datasets: Dataset[];
  isLoading: boolean;
}

export default function ProcessingQueue({ datasets, isLoading }: ProcessingQueueProps) {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'downloading':
        return (
          <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">
            Downloading
          </Badge>
        );
      case 'structuring':
        return (
          <Badge className="bg-purple-100 text-purple-800 hover:bg-purple-100">
            Structuring
          </Badge>
        );
      case 'generating':
        return (
          <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">
            Generating Metadata
          </Badge>
        );
      default:
        return (
          <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-100">
            {status}
          </Badge>
        );
    }
  };
  
  const getFileIcon = (title: string) => {
    if (title.toLowerCase().includes('excel') || title.toLowerCase().includes('spreadsheet')) {
      return (
        <div className="flex-shrink-0 h-10 w-10 bg-indigo-100 rounded-full flex items-center justify-center">
          <svg className="h-5 w-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
      );
    } else if (title.toLowerCase().includes('genom') || title.toLowerCase().includes('sequenc')) {
      return (
        <div className="flex-shrink-0 h-10 w-10 bg-blue-100 rounded-full flex items-center justify-center">
          <svg className="h-5 w-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        </div>
      );
    } else if (title.toLowerCase().includes('social') || title.toLowerCase().includes('media')) {
      return (
        <div className="flex-shrink-0 h-10 w-10 bg-green-100 rounded-full flex items-center justify-center">
          <svg className="h-5 w-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
          </svg>
        </div>
      );
    } else {
      return (
        <div className="flex-shrink-0 h-10 w-10 bg-primary-100 rounded-full flex items-center justify-center">
          <svg className="h-5 w-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
        </div>
      );
    }
  };
  
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Processing Queue</h2>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" className="flex items-center">
            <Pause className="h-4 w-4 mr-1" />
            Pause All
          </Button>
          <Button size="sm" className="flex items-center">
            <PlusCircle className="h-4 w-4 mr-1" />
            Add Dataset
          </Button>
        </div>
      </div>
      
      {isLoading ? (
        <div className="animate-pulse space-y-4">
          <div className="h-10 bg-gray-200 rounded"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
        </div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-10">
          <p className="text-gray-500">No datasets currently processing.</p>
          <p className="text-gray-400 text-sm">Add a dataset to begin processing.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Progress</TableHead>
                <TableHead>ETC</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {datasets.map((dataset) => (
                <TableRow key={dataset.id}>
                  <TableCell>
                    <div className="flex items-center">
                      {getFileIcon(dataset.title)}
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">{dataset.title}</div>
                        <div className="text-sm text-gray-500">{dataset.size}</div>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm text-gray-900">{dataset.source}</div>
                    <div className="text-xs text-gray-500">{new URL(dataset.sourceUrl).hostname}</div>
                  </TableCell>
                  <TableCell>
                    {getStatusBadge(dataset.status)}
                  </TableCell>
                  <TableCell>
                    <ProgressBar 
                      value={dataset.progress || 0} 
                      showPercentage={true}
                      color={
                        dataset.status === 'downloading' ? 'amber' :
                        dataset.status === 'structuring' ? 'purple' :
                        dataset.status === 'generating' ? 'blue' : 'primary'
                      }
                    />
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-sm text-gray-500">
                    {dataset.estimatedTimeToCompletion || '—'}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-right text-sm font-medium">
                    <Button variant="ghost" size="sm" className="text-primary-600 hover:text-primary-900 mr-3">
                      <Pause className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-900">
                      <X className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
