import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { PlusIcon, BellIcon, DatabaseIcon } from "lucide-react";

const Header = () => {
  const [location] = useLocation();

  const isActive = (path: string) => {
    return location === path;
  };

  return (
    <header className="bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <DatabaseIcon className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">DataMeta AI</span>
            </div>
            <nav className="hidden md:ml-8 md:flex md:space-x-8">
              <Link href="/">
                <a className={`px-1 pt-1 border-b-2 text-sm font-medium ${isActive("/") ? "text-primary-600 border-primary-500" : "text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300"}`}>
                  Dashboard
                </a>
              </Link>
              <Link href="/datasets">
                <a className={`px-1 pt-1 border-b-2 text-sm font-medium ${isActive("/datasets") ? "text-primary-600 border-primary-500" : "text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300"}`}>
                  Datasets
                </a>
              </Link>
              <Link href="/documentation">
                <a className={`px-1 pt-1 border-b-2 text-sm font-medium ${isActive("/documentation") ? "text-primary-600 border-primary-500" : "text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300"}`}>
                  Documentation
                </a>
              </Link>
              <Link href="/api">
                <a className={`px-1 pt-1 border-b-2 text-sm font-medium ${isActive("/api") ? "text-primary-600 border-primary-500" : "text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300"}`}>
                  API
                </a>
              </Link>
            </nav>
          </div>
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Button className="relative inline-flex items-center">
                <PlusIcon className="-ml-1 mr-2 h-4 w-4" />
                <span>New Project</span>
              </Button>
            </div>
            <div className="ml-4 flex items-center md:ml-6">
              <button className="p-1 rounded-full text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
                <BellIcon className="h-6 w-6" />
              </button>
              <div className="ml-3 relative">
                <div>
                  <Avatar>
                    <AvatarFallback>JD</AvatarFallback>
                  </Avatar>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
