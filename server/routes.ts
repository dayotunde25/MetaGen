import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertDatasetSchema, insertMetadataSchema, insertProcessingHistorySchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Get dashboard statistics
  app.get("/api/statistics", async (req, res) => {
    try {
      const stats = await storage.getProcessingStatistics();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch statistics",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get all datasets
  app.get("/api/datasets", async (req, res) => {
    try {
      const datasets = await storage.getAllDatasets();
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch datasets",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get recent datasets
  app.get("/api/datasets/recent", async (req, res) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 3;
      const datasets = await storage.getRecentDatasets(limit);
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch recent datasets",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Search datasets
  app.get("/api/datasets/search", async (req, res) => {
    try {
      const query = req.query.q as string || "";
      const format = req.query.format as string;
      const category = req.query.category as string;
      const dateRange = req.query.dateRange as string;
      
      const filters = {
        format,
        category,
        dateRange
      };
      
      const datasets = await storage.searchDatasets(query, filters);
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to search datasets",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get datasets by status
  app.get("/api/datasets/status/:status", async (req, res) => {
    try {
      const status = req.params.status;
      const datasets = await storage.getDatasetsByStatus(status);
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch datasets by status",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get dataset by ID
  app.get("/api/datasets/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ message: "Invalid dataset ID" });
      }
      
      const dataset = await storage.getDataset(id);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      res.json(dataset);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch dataset",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Create new dataset
  app.post("/api/datasets", async (req, res) => {
    try {
      const validatedData = insertDatasetSchema.parse(req.body);
      const dataset = await storage.createDataset(validatedData);
      
      // Create initial processing history record
      await storage.createProcessingHistory({
        datasetId: dataset.id,
        operation: "download",
        status: "in_progress",
        details: `Started downloading dataset from ${dataset.source}`,
        startTime: new Date()
      });
      
      res.status(201).json(dataset);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ 
          message: "Invalid dataset data", 
          errors: error.errors 
        });
      }
      
      res.status(500).json({ 
        message: "Failed to create dataset",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Update dataset status
  app.patch("/api/datasets/:id/status", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ message: "Invalid dataset ID" });
      }
      
      const { status, progress, estimatedTimeToCompletion } = req.body;
      if (!status) {
        return res.status(400).json({ message: "Status is required" });
      }
      
      const updatedDataset = await storage.updateDatasetStatus(
        id, 
        status, 
        progress, 
        estimatedTimeToCompletion
      );
      
      if (!updatedDataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      res.json(updatedDataset);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to update dataset status",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get metadata for a dataset
  app.get("/api/datasets/:id/metadata", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      if (isNaN(datasetId)) {
        return res.status(400).json({ message: "Invalid dataset ID" });
      }
      
      const metadata = await storage.getMetadataByDatasetId(datasetId);
      if (!metadata) {
        return res.status(404).json({ message: "Metadata not found for this dataset" });
      }
      
      res.json(metadata);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch metadata",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Create metadata for a dataset
  app.post("/api/metadata", async (req, res) => {
    try {
      const validatedData = insertMetadataSchema.parse(req.body);
      const metadata = await storage.createMetadata(validatedData);
      
      res.status(201).json(metadata);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ 
          message: "Invalid metadata", 
          errors: error.errors 
        });
      }
      
      res.status(500).json({ 
        message: "Failed to create metadata",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Get processing history for a dataset
  app.get("/api/datasets/:id/history", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      if (isNaN(datasetId)) {
        return res.status(400).json({ message: "Invalid dataset ID" });
      }
      
      const history = await storage.getProcessingHistoryByDatasetId(datasetId);
      res.json(history);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch processing history",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Create processing history entry
  app.post("/api/history", async (req, res) => {
    try {
      const validatedData = insertProcessingHistorySchema.parse(req.body);
      const history = await storage.createProcessingHistory(validatedData);
      
      res.status(201).json(history);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ 
          message: "Invalid processing history data", 
          errors: error.errors 
        });
      }
      
      res.status(500).json({ 
        message: "Failed to create processing history entry",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Update processing history entry
  app.patch("/api/history/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ message: "Invalid history entry ID" });
      }
      
      const { status, details } = req.body;
      if (!status) {
        return res.status(400).json({ message: "Status is required" });
      }
      
      const endTime = status === 'success' || status === 'failed' ? new Date() : undefined;
      const updatedHistory = await storage.updateProcessingHistory(id, status, details, endTime);
      
      if (!updatedHistory) {
        return res.status(404).json({ message: "Processing history entry not found" });
      }
      
      res.json(updatedHistory);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to update processing history entry",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  // Implement semantic search for datasets
  app.get("/api/semantic-search", async (req, res) => {
    try {
      const query = req.query.q as string || "";
      if (!query.trim()) {
        return res.status(400).json({ message: "Search query is required" });
      }
      
      // For MVP, we'll use basic text search and simulate semantic capabilities
      // In a real implementation, this would use proper NLP library
      const searchResults = await storage.searchDatasets(query);
      
      // Add a relevance score to simulate NLP based ranking
      const results = searchResults.map(dataset => ({
        ...dataset,
        relevanceScore: Math.random() * 0.5 + 0.5 // Random score between 0.5 and 1
      }));
      
      // Sort by simulated relevance
      results.sort((a, b) => b.relevanceScore - a.relevanceScore);
      
      res.json(results);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to perform semantic search",
        error: error instanceof Error ? error.message : "Unknown error" 
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
