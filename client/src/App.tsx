import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import Datasets from "@/pages/datasets";
import Documentation from "@/pages/documentation";
import Api from "@/pages/api";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/datasets" component={Datasets} />
      <Route path="/documentation" component={Documentation} />
      <Route path="/api" component={Api} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex flex-col min-h-screen">
        <Header />
        <div className="flex-grow">
          <Router />
        </div>
        <Footer />
      </div>
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
