import { apiRequest } from "./queryClient";
import { Dataset, InsertDataset, ProcessingQueue, InsertProcessingQueue } from "@shared/schema";

// Dataset API functions
export const fetchDatasets = async (): Promise<Dataset[]> => {
  const res = await apiRequest("GET", "/api/datasets", undefined);
  return res.json();
};

export const fetchDataset = async (id: number): Promise<Dataset> => {
  const res = await apiRequest("GET", `/api/datasets/${id}`, undefined);
  return res.json();
};

export const searchDatasets = async (query: string, filters?: Record<string, any>): Promise<Dataset[]> => {
  const params = new URLSearchParams();
  if (query) params.append("q", query);
  if (filters) {
    Object.entries(filters).forEach(([key, value]) => {
      if (value) params.append(key, value.toString());
    });
  }
  
  const res = await apiRequest("GET", `/api/datasets/search?${params.toString()}`, undefined);
  return res.json();
};

export const addDataset = async (dataset: InsertDataset): Promise<Dataset> => {
  const res = await apiRequest("POST", "/api/datasets", dataset);
  return res.json();
};

export const updateDataset = async (id: number, dataset: Partial<Dataset>): Promise<Dataset> => {
  const res = await apiRequest("PATCH", `/api/datasets/${id}`, dataset);
  return res.json();
};

export const deleteDataset = async (id: number): Promise<void> => {
  await apiRequest("DELETE", `/api/datasets/${id}`, undefined);
};

export const downloadDataset = async (id: number): Promise<Blob> => {
  const res = await apiRequest("GET", `/api/datasets/${id}/download`, undefined);
  return res.blob();
};

// Processing Queue API functions
export const fetchProcessingQueue = async (): Promise<ProcessingQueue[]> => {
  const res = await apiRequest("GET", "/api/processing", undefined);
  return res.json();
};

export const fetchProcessingHistory = async (): Promise<ProcessingQueue[]> => {
  const res = await apiRequest("GET", "/api/processing/history", undefined);
  return res.json();
};

export const addToProcessingQueue = async (item: InsertProcessingQueue): Promise<ProcessingQueue> => {
  const res = await apiRequest("POST", "/api/processing", item);
  return res.json();
};

export const updateProcessingItem = async (id: number, data: Partial<ProcessingQueue>): Promise<ProcessingQueue> => {
  const res = await apiRequest("PATCH", `/api/processing/${id}`, data);
  return res.json();
};

export const removeFromProcessingQueue = async (id: number): Promise<void> => {
  await apiRequest("DELETE", `/api/processing/${id}`, undefined);
};

// Repository API functions
export const fetchRepositories = async (): Promise<string[]> => {
  const res = await apiRequest("GET", "/api/repositories", undefined);
  return res.json();
};

export const searchRepository = async (repository: string, query: string): Promise<any[]> => {
  const res = await apiRequest("GET", `/api/repositories/${repository}/search?q=${encodeURIComponent(query)}`, undefined);
  return res.json();
};

// Categories API functions
export const fetchCategories = async (): Promise<{name: string, count: number, icon: string, color: string}[]> => {
  const res = await apiRequest("GET", "/api/categories", undefined);
  return res.json();
};
