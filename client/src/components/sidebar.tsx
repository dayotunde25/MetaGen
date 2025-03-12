import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";

export default function Sidebar() {
  const [location] = useLocation();

  const navItems = [
    { name: "Dashboard", path: "/", icon: "dashboard" },
    { name: "Search Datasets", path: "/search", icon: "search" },
    { name: "My Datasets", path: "/my-datasets", icon: "folder" },
    { name: "Settings", path: "/settings", icon: "settings" },
  ];

  return (
    <aside className="w-64 h-full bg-white shadow-md z-10 flex-shrink-0 hidden md:block">
      <div className="p-4 border-b border-neutral-light">
        <h1 className="font-bold text-2xl text-primary">DataMetaGen</h1>
        <p className="text-sm text-neutral-medium">AI Research Metadata Generator</p>
      </div>
      
      <nav className="py-4">
        <ul>
          {navItems.map((item) => (
            <li key={item.path} className="mb-1">
              <Link href={item.path}>
                <a
                  className={cn(
                    "flex items-center px-4 py-3 hover:bg-primary-light hover:bg-opacity-10 hover:text-primary transition-all",
                    location === item.path && "bg-primary-light bg-opacity-10 text-primary border-l-4 border-primary"
                  )}
                >
                  <span className="material-icons mr-3">{item.icon}</span>
                  {item.name}
                </a>
              </Link>
            </li>
          ))}
          
          <li className="mb-1 mt-8">
            <Link href="/help">
              <a className="flex items-center px-4 py-3 hover:bg-primary-light hover:bg-opacity-10 hover:text-primary transition-all">
                <span className="material-icons mr-3">help</span>
                Help & Documentation
              </a>
            </Link>
          </li>
        </ul>
      </nav>
    </aside>
  );
}
