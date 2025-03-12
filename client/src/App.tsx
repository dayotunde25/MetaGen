import { Route, Switch } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Discover from "@/pages/Discover";
import SourceDataset from "@/pages/SourceDataset";
import ProcessingQueue from "@/pages/ProcessingQueue";
import Library from "@/pages/Library";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { useState } from "react";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Discover} />
      <Route path="/source" component={SourceDataset} />
      <Route path="/process" component={ProcessingQueue} />
      <Route path="/library" component={Library} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex-1 flex flex-col min-h-screen">
        <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
        <div className="flex-1 flex flex-col min-h-screen">
          <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
          <main className="flex-1 overflow-y-auto bg-slate-50 p-4 sm:p-6 lg:p-8">
            <Router />
          </main>
        </div>
      </div>
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
