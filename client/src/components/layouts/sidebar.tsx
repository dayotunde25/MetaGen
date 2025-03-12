import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { 
  Database, 
  Search, 
  FolderDown, 
  FileText, 
  History, 
  ArrowLeftToLine, 
  ArrowRightToLine, 
  Menu, 
  User
} from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
}

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
}

export default function Sidebar({ isOpen, toggleSidebar }: SidebarProps) {
  const [location] = useLocation();
  
  const navItems: NavItem[] = [
    { href: "/", label: "Dashboard", icon: <Database className="h-5 w-5" /> },
    { href: "/search", label: "Search Datasets", icon: <Search className="h-5 w-5" /> },
    { href: "/downloaded", label: "Downloaded Datasets", icon: <FolderDown className="h-5 w-5" /> },
    { href: "/metadata", label: "Generated Metadata", icon: <FileText className="h-5 w-5" /> },
    { href: "/history", label: "Processing History", icon: <History className="h-5 w-5" /> },
  ];
  
  return (
    <div 
      className={cn(
        "fixed z-30 inset-y-0 left-0 bg-gray-800 text-white transition-all duration-300 transform shadow-lg md:relative md:translate-x-0 flex flex-col",
        isOpen ? "translate-x-0 md:w-64" : "-translate-x-full md:translate-x-0 md:w-20"
      )}
    >
      {/* Logo & Toggle */}
      <div className="flex items-center justify-between p-4 h-16 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <Database className="h-6 w-6 text-primary-500" />
          <span className={cn("font-semibold text-lg", !isOpen && "md:hidden")}>DataMeta</span>
        </div>
        <button onClick={toggleSidebar} className="p-1 rounded-md hover:bg-gray-700 md:hidden">
          <Menu className="h-5 w-5" />
        </button>
        <button onClick={toggleSidebar} className="p-1 rounded-md hover:bg-gray-700 hidden md:block">
          {isOpen ? (
            <ArrowLeftToLine className="h-5 w-5" />
          ) : (
            <ArrowRightToLine className="h-5 w-5" />
          )}
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <Link 
            key={item.href} 
            href={item.href}
          >
            <a
              className={cn(
                "flex items-center px-3 py-2 text-sm rounded-md",
                location === item.href 
                  ? "bg-gray-900 text-white" 
                  : "text-gray-300 hover:bg-gray-700 hover:text-white"
              )}
            >
              <span className="mr-3">{item.icon}</span>
              <span className={cn(!isOpen && "md:hidden")}>{item.label}</span>
            </a>
          </Link>
        ))}
      </nav>
      
      {/* User Profile */}
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0 h-9 w-9 rounded-full bg-gray-600 flex items-center justify-center">
            <User className="h-5 w-5" />
          </div>
          <div className={cn(!isOpen && "md:hidden")}>
            <div className="text-sm font-medium">Research Admin</div>
            <div className="text-xs text-gray-400">admin@research.org</div>
          </div>
        </div>
      </div>
    </div>
  );
}
