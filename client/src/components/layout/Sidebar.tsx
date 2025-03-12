import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { 
  Search, CloudDownload, Cog, Database, X, CheckCircle, AlertCircle, Clock
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchProcessingQueue } from "@/lib/api";

interface SidebarProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

export default function Sidebar({ open, setOpen }: SidebarProps) {
  const [location] = useLocation();
  
  // Fetch processing queue
  const { data: processingQueue } = useQuery({
    queryKey: ['/api/processing'],
    staleTime: 10000 // Refresh every 10 seconds
  });
  
  const queueCount = processingQueue?.length || 0;

  return (
    <div 
      className={cn(
        "fixed z-30 inset-y-0 left-0 w-64 transition duration-300 transform bg-white border-r border-slate-200 lg:translate-x-0 lg:static lg:w-80",
        open ? "translate-x-0" : "-translate-x-full"
      )}
    >
      <div className="flex items-center justify-between h-16 px-6 border-b border-slate-200">
        <div className="flex items-center">
          <svg className="h-8 w-8 text-primary-600" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
          </svg>
          <span className="ml-2 text-xl font-semibold text-primary-700">MetaDataset</span>
        </div>
        <button onClick={() => setOpen(false)} className="lg:hidden">
          <X className="h-6 w-6 text-slate-500" />
        </button>
      </div>

      <nav className="mt-6 px-4">
        <div className="space-y-1">
          <Link href="/">
            <a 
              className={cn(
                "w-full flex items-center px-4 py-3 rounded-lg transition",
                location === "/" 
                  ? "bg-primary-50 text-primary-700" 
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              <Search className="h-5 w-5" />
              <span className="ml-3 text-base font-medium">Discover Datasets</span>
            </a>
          </Link>
          <Link href="/source">
            <a 
              className={cn(
                "w-full flex items-center px-4 py-3 rounded-lg transition",
                location === "/source" 
                  ? "bg-primary-50 text-primary-700" 
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              <CloudDownload className="h-5 w-5" />
              <span className="ml-3 text-base font-medium">Source New Dataset</span>
            </a>
          </Link>
          <Link href="/process">
            <a 
              className={cn(
                "w-full flex items-center px-4 py-3 rounded-lg transition",
                location === "/process" 
                  ? "bg-primary-50 text-primary-700" 
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              <Cog className="h-5 w-5" />
              <span className="ml-3 text-base font-medium">Processing Queue</span>
              {queueCount > 0 && (
                <span className="ml-auto bg-amber-500 text-white text-xs font-semibold px-2 py-1 rounded-full">
                  {queueCount}
                </span>
              )}
            </a>
          </Link>
          <Link href="/library">
            <a 
              className={cn(
                "w-full flex items-center px-4 py-3 rounded-lg transition",
                location === "/library" 
                  ? "bg-primary-50 text-primary-700" 
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              <Database className="h-5 w-5" />
              <span className="ml-3 text-base font-medium">My Dataset Library</span>
            </a>
          </Link>
        </div>
        
        <div className="mt-8">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Recent Activity</h3>
          <div className="mt-3 space-y-3">
            <RecentActivityItem 
              type="success" 
              text="COVID-19 Dataset processed"
              time="2h ago"
            />
            <RecentActivityItem 
              type="pending" 
              text="Climate Data queued"
              time="5h ago"
            />
            <RecentActivityItem 
              type="info" 
              text="Added Economic Dataset"
              time="Yesterday"
            />
          </div>
        </div>
      </nav>
    </div>
  );
}

interface RecentActivityItemProps {
  type: "success" | "error" | "pending" | "info";
  text: string;
  time: string;
}

function RecentActivityItem({ type, text, time }: RecentActivityItemProps) {
  const getIcon = () => {
    switch (type) {
      case "success":
        return <div className="w-2 h-2 bg-green-500 rounded-full"></div>;
      case "error":
        return <div className="w-2 h-2 bg-red-500 rounded-full"></div>;
      case "pending":
        return <div className="w-2 h-2 bg-amber-500 rounded-full"></div>;
      case "info":
        return <div className="w-2 h-2 bg-primary-500 rounded-full"></div>;
      default:
        return null;
    }
  };

  return (
    <div className="flex items-center px-4 py-2 text-sm">
      {getIcon()}
      <span className="ml-3 text-slate-600">{text}</span>
      <span className="ml-auto text-xs text-slate-400">{time}</span>
    </div>
  );
}
