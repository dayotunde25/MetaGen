import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Bell, Cog, Search, Menu, SlidersHorizontal } from "lucide-react";
import { Button } from "../ui/button";
import AdvancedSearch from "../datasets/AdvancedSearch";

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

export default function Header({ sidebarOpen, setSidebarOpen }: HeaderProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);

  return (
    <header className="z-10 py-4 bg-white shadow-sm lg:static lg:overflow-y-visible">
      <div className="px-4 sm:px-6 lg:px-8 flex justify-between items-center">
        <button onClick={() => setSidebarOpen(true)} className="text-slate-500 focus:outline-none lg:hidden">
          <Menu className="h-6 w-6" />
        </button>
        
        <div className="relative w-full max-w-md mx-4 hidden sm:block">
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            type="text"
            className="w-full rounded-lg border border-slate-300 bg-white pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            placeholder="Search datasets..."
          />
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400">
            <Search className="h-4 w-4" />
          </div>
          <button 
            onClick={() => setShowAdvancedSearch(!showAdvancedSearch)} 
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-primary-500"
          >
            <SlidersHorizontal className="h-4 w-4" />
          </button>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="p-1 rounded-full text-slate-400 hover:text-slate-500">
            <Bell className="h-5 w-5" />
          </button>
          <button className="p-1 rounded-full text-slate-400 hover:text-slate-500">
            <Cog className="h-5 w-5" />
          </button>
          <div className="flex items-center">
            <span className="hidden md:block text-sm font-medium mr-2">Jane Researcher</span>
            <div className="h-8 w-8 rounded-full bg-primary-600 flex items-center justify-center text-white">
              <span className="text-sm font-medium">JR</span>
            </div>
          </div>
        </div>
      </div>
      
      {showAdvancedSearch && <AdvancedSearch onClose={() => setShowAdvancedSearch(false)} />}
    </header>
  );
}
