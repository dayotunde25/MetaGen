import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import SearchDatasets from "@/pages/search-datasets";
import DownloadedDatasets from "@/pages/downloaded-datasets";
import GeneratedMetadata from "@/pages/generated-metadata";
import ProcessingHistory from "@/pages/processing-history";
import AppShell from "@/components/layouts/app-shell";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/search" component={SearchDatasets} />
      <Route path="/downloaded" component={DownloadedDatasets} />
      <Route path="/metadata" component={GeneratedMetadata} />
      <Route path="/history" component={ProcessingHistory} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>
        <Router />
      </AppShell>
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
