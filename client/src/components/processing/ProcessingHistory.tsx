import { ProcessingQueue as ProcessingQueueType } from "@shared/schema";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getStatusColor } from "@/lib/utils";
import { CheckCircle, AlertCircle, RefreshCw } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchProcessingHistory } from "@/lib/api";

export default function ProcessingHistory() {
  const { data: historyItems = [], isLoading } = useQuery({
    queryKey: ['/api/processing/history'],
  });

  if (isLoading) {
    return <p>Loading history...</p>;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-200">
        <h2 className="font-semibold">Processing History</h2>
      </div>
      {historyItems.length === 0 ? (
        <div className="px-6 py-8 text-center text-slate-500">
          No processing history available
        </div>
      ) : (
        <div className="divide-y divide-slate-200">
          {historyItems.map((item) => (
            <div key={item.id} className="px-6 py-4">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="font-medium text-slate-900">{item.name}</h3>
                  <p className="text-sm text-slate-500">
                    Completed {new Date(item.createdAt).toLocaleDateString()} at {new Date(item.createdAt).toLocaleTimeString()}
                  </p>
                </div>
                <div className="flex items-center space-x-2 mt-2 md:mt-0">
                  {item.status === "completed" || item.status === "successful" ? (
                    <>
                      <Badge className="bg-green-100 text-green-800">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Successful
                      </Badge>
                      {item.progress === 100 && (
                        <span className="text-sm text-slate-500">FAIR Score: 95%</span>
                      )}
                    </>
                  ) : (
                    <>
                      <Badge className="bg-red-100 text-red-800">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        Failed
                      </Badge>
                      <Button 
                        variant="link" 
                        className="text-primary-600 hover:text-primary-700 text-sm font-medium p-0"
                      >
                        Retry
                      </Button>
                    </>
                  )}
                </div>
              </div>
              
              {item.error && (
                <div className="mt-2 p-2 bg-red-50 border border-red-100 rounded text-sm text-red-700">
                  Error: {item.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 text-center">
        <Button variant="link" className="text-primary-600 hover:text-primary-700 text-sm font-medium">
          View complete history
        </Button>
      </div>
    </div>
  );
}
