import { useState } from "react";
import { Menu, Bell, Settings } from "lucide-react";

interface TopNavProps {
  toggleSidebar: () => void;
}

export default function TopNav({ toggleSidebar }: TopNavProps) {
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  return (
    <header className="bg-white shadow-sm z-10">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <button 
              onClick={toggleSidebar} 
              className="p-2 rounded-md text-gray-500 md:hidden"
            >
              <Menu className="h-5 w-5" />
            </button>
            <h1 className="text-xl font-semibold text-gray-800 ml-2 md:ml-0">
              AI Research Metadata Generator
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="relative">
              <button 
                onClick={() => setNotificationsOpen(!notificationsOpen)} 
                className="p-2 rounded-full text-gray-500 hover:bg-gray-100 relative"
              >
                <Bell className="h-5 w-5" />
                <span className="absolute top-1 right-1 bg-primary-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                  3
                </span>
              </button>
              {notificationsOpen && (
                <div 
                  className="origin-top-right absolute right-0 mt-2 w-72 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5"
                  onClick={() => setNotificationsOpen(false)}
                >
                  <div className="py-1 divide-y divide-gray-100">
                    <div className="px-4 py-3">
                      <p className="text-sm">Dataset processing complete for <span className="font-medium">COVID-19 Research</span></p>
                      <p className="text-xs text-gray-500 mt-1">1h ago</p>
                    </div>
                    <div className="px-4 py-3">
                      <p className="text-sm">New dataset available: <span className="font-medium">Climate Change Metrics</span></p>
                      <p className="text-xs text-gray-500 mt-1">3h ago</p>
                    </div>
                    <div className="px-4 py-3">
                      <p className="text-sm">Metadata generation failed for <span className="font-medium">Genome Sequencing</span></p>
                      <p className="text-xs text-gray-500 mt-1">5h ago</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="relative">
              <button 
                onClick={() => setSettingsOpen(!settingsOpen)} 
                className="p-2 rounded-full text-gray-500 hover:bg-gray-100"
              >
                <Settings className="h-5 w-5" />
              </button>
              {settingsOpen && (
                <div 
                  className="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5"
                  onClick={() => setSettingsOpen(false)}
                >
                  <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Account Settings
                  </a>
                  <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    API Configuration
                  </a>
                  <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Storage Options
                  </a>
                  <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Sign out
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
