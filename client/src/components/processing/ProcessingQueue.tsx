import { ProcessingQueue as ProcessingQueueType } from "@shared/schema";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getProgressColor, getStatusColor } from "@/lib/utils";
import { Pause, X } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchProcessingQueue, removeFromProcessingQueue } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";

export default function ProcessingQueue() {
  const { data: processingItems = [], isLoading } = useQuery({
    queryKey: ['/api/processing'],
    staleTime: 10000 // Refresh every 10 seconds
  });
  
  const handleRemoveItem = async (id: number) => {
    try {
      await removeFromProcessingQueue(id);
      queryClient.invalidateQueries({ queryKey: ['/api/processing'] });
    } catch (error) {
      console.error("Failed to remove item from queue", error);
    }
  };

  if (isLoading) {
    return <p>Loading queue...</p>;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden mb-6">
      <div className="px-6 py-4 border-b border-slate-200">
        <h2 className="font-semibold">
          Active Processing ({processingItems.length})
        </h2>
      </div>
      {processingItems.length === 0 ? (
        <div className="px-6 py-8 text-center text-slate-500">
          No datasets currently being processed
        </div>
      ) : (
        <div className="divide-y divide-slate-200">
          {processingItems.map((item) => (
            <div key={item.id} className="px-6 py-4">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-3">
                <div>
                  <h3 className="font-medium text-slate-900">{item.name}</h3>
                  <p className="text-sm text-slate-500">
                    Sourced from {item.source} {item.size && `- ${item.size}`}
                  </p>
                </div>
                <div className="flex items-center mt-2 md:mt-0">
                  <Badge 
                    className={`mr-2 ${getStatusColor(item.status)}`}
                  >
                    {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
                  </Badge>
                  {item.status === "processing" && (
                    <button className="text-slate-400 hover:text-slate-500">
                      <Pause className="h-4 w-4" />
                    </button>
                  )}
                  {item.status === "queued" && (
                    <button 
                      className="text-slate-400 hover:text-slate-500"
                      onClick={() => handleRemoveItem(item.id)}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
              <Progress
                value={item.progress}
                className="w-full h-2 mb-2"
                color={getProgressColor(item.progress)}
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>{item.progress}% complete</span>
                <span>
                  {item.status === "queued" 
                    ? `Estimated start: ${item.estimatedCompletionTime || 'Unknown'}`
                    : `Estimated time remaining: ${item.estimatedCompletionTime || 'Unknown'}`
                  }
                </span>
              </div>
              
              {item.status === "processing" && (
                <div className="mt-3 space-y-1">
                  <div className="flex items-center text-xs text-slate-500">
                    <div className="w-24">Download:</div>
                    <div className="w-16 text-center">
                      {item.progress > 20 ? '100%' : Math.min(100, item.progress * 5) + '%'}
                    </div>
                    <div className="flex-grow h-1.5 bg-slate-100 rounded-full overflow-hidden ml-2">
                      <div 
                        className="h-full bg-green-500 rounded-full" 
                        style={{ width: item.progress > 20 ? '100%' : `${Math.min(100, item.progress * 5)}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex items-center text-xs text-slate-500">
                    <div className="w-24">Structuring:</div>
                    <div className="w-16 text-center">
                      {item.progress > 60 ? '100%' : item.progress > 20 ? Math.min(100, (item.progress - 20) * 2.5) + '%' : '0%'}
                    </div>
                    <div className="flex-grow h-1.5 bg-slate-100 rounded-full overflow-hidden ml-2">
                      <div 
                        className="h-full bg-green-500 rounded-full" 
                        style={{ 
                          width: item.progress > 60 
                            ? '100%' 
                            : item.progress > 20 
                              ? `${Math.min(100, (item.progress - 20) * 2.5)}%` 
                              : '0%' 
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex items-center text-xs text-slate-500">
                    <div className="w-24">Metadata:</div>
                    <div className="w-16 text-center">
                      {item.progress > 80 ? Math.min(100, (item.progress - 60) * 5) + '%' : '0%'}
                    </div>
                    <div className="flex-grow h-1.5 bg-slate-100 rounded-full overflow-hidden ml-2">
                      <div 
                        className="h-full bg-green-500 rounded-full" 
                        style={{ 
                          width: item.progress > 60 
                            ? `${Math.min(100, (item.progress - 60) * 2.5)}%` 
                            : '0%' 
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex items-center text-xs text-slate-500">
                    <div className="w-24">FAIR Score:</div>
                    <div className="w-16 text-center">
                      {item.progress > 80 ? Math.min(100, (item.progress - 80) * 5) + '%' : '--'}
                    </div>
                    <div className="flex-grow h-1.5 bg-slate-100 rounded-full overflow-hidden ml-2">
                      <div 
                        className="h-full bg-slate-300 rounded-full" 
                        style={{ 
                          width: item.progress > 80 
                            ? `${Math.min(100, (item.progress - 80) * 5)}%` 
                            : '0%' 
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
              
              {item.error && (
                <div className="mt-3 p-2 bg-red-50 border border-red-100 rounded text-sm text-red-700">
                  Error: {item.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
