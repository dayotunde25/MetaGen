import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Menu } from "lucide-react";
import Sidebar from "./sidebar";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";

export default function Navbar() {
  const [location] = useLocation();
  const [open, setOpen] = useState(false);
  
  // Get page title based on current route
  const getPageTitle = () => {
    switch (location) {
      case "/":
        return "Dashboard";
      case "/search":
        return "Search Datasets";
      case "/my-datasets":
        return "My Datasets";
      case "/add-dataset":
        return "Add Dataset";
      default:
        if (location.startsWith("/dataset/")) {
          return "Dataset Details";
        }
        return "DataMetaGen";
    }
  };

  return (
    <header className="bg-white shadow-sm flex items-center justify-between p-4 md:px-8">
      <div className="flex items-center">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden mr-4">
              <Menu className="h-6 w-6 text-neutral-dark" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-64">
            <Sidebar />
          </SheetContent>
        </Sheet>
        <h2 className="font-semibold text-lg md:text-xl">{getPageTitle()}</h2>
      </div>
      
      <div className="flex items-center">
        <div className="relative mr-2">
          <Button variant="ghost" size="icon" aria-label="Notifications">
            <span className="material-icons">notifications</span>
            <span className="absolute top-1 right-1 bg-destructive rounded-full w-2 h-2"></span>
          </Button>
        </div>
        
        <div className="flex items-center ml-4">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white font-medium mr-2">
            JD
          </div>
          <span className="hidden md:block">John Doe</span>
        </div>
      </div>
    </header>
  );
}
