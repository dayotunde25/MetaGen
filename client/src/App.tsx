import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import SearchDatasets from "@/pages/search-datasets";
import MyDatasets from "@/pages/my-datasets";
import DatasetDetails from "@/pages/dataset-details";
import AddDataset from "@/pages/add-dataset";
import Sidebar from "@/components/sidebar";
import Navbar from "@/components/navbar";

function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto bg-neutral-lightest p-4 md:p-8">
          {children}
        </main>
      </div>
    </div>
  );
}

function Router() {
  return (
    <Switch>
      <Route path="/">
        <MainLayout>
          <Dashboard />
        </MainLayout>
      </Route>
      <Route path="/search">
        <MainLayout>
          <SearchDatasets />
        </MainLayout>
      </Route>
      <Route path="/my-datasets">
        <MainLayout>
          <MyDatasets />
        </MainLayout>
      </Route>
      <Route path="/dataset/:id">
        {(params) => (
          <MainLayout>
            <DatasetDetails id={parseInt(params.id)} />
          </MainLayout>
        )}
      </Route>
      <Route path="/add-dataset">
        <MainLayout>
          <AddDataset />
        </MainLayout>
      </Route>
      <Route>
        <NotFound />
      </Route>
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
