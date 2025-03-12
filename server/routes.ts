import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertDatasetSchema, insertMetadataSchema } from "@shared/schema";
import { SemanticSearch } from "./nlp/semantic-search";
import { DatasetService } from "./services/dataset-service";
import { MetadataService } from "./services/metadata-service";

export async function registerRoutes(app: Express): Promise<Server> {
  // Initialize services
  const semanticSearch = new SemanticSearch();
  const datasetService = new DatasetService(storage);
  const metadataService = new MetadataService(storage);

  // API routes
  // Get all datasets
  app.get("/api/datasets", async (req, res) => {
    try {
      const datasets = await storage.getAllDatasets();
      res.json({ datasets });
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch datasets" });
    }
  });

  // Get dataset by ID
  app.get("/api/datasets/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const dataset = await storage.getDataset(id);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      res.json({ dataset });
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch dataset" });
    }
  });

  // Create new dataset
  app.post("/api/datasets", async (req, res) => {
    try {
      const validatedData = insertDatasetSchema.parse(req.body);
      const dataset = await storage.createDataset(validatedData);
      res.status(201).json({ dataset });
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ error: error.errors });
      }
      res.status(500).json({ error: "Failed to create dataset" });
    }
  });

  // Update dataset
  app.patch("/api/datasets/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const dataset = await storage.getDataset(id);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      const validatedData = insertDatasetSchema.partial().parse(req.body);
      const updatedDataset = await storage.updateDataset(id, validatedData);
      
      res.json({ dataset: updatedDataset });
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ error: error.errors });
      }
      res.status(500).json({ error: "Failed to update dataset" });
    }
  });

  // Delete dataset
  app.delete("/api/datasets/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const dataset = await storage.getDataset(id);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      await storage.deleteDataset(id);
      res.status(204).send();
    } catch (error) {
      res.status(500).json({ error: "Failed to delete dataset" });
    }
  });

  // Get metadata for dataset
  app.get("/api/datasets/:id/metadata", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      const dataset = await storage.getDataset(datasetId);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      const metadata = await storage.getMetadataByDatasetId(datasetId);
      
      if (!metadata) {
        return res.status(404).json({ error: "Metadata not found for this dataset" });
      }
      
      res.json({ metadata });
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch metadata" });
    }
  });

  // Create or update metadata for dataset
  app.post("/api/datasets/:id/metadata", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      const dataset = await storage.getDataset(datasetId);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      // Add dataset ID to the request body
      const metadataData = { ...req.body, datasetId };
      const validatedData = insertMetadataSchema.parse(metadataData);
      
      // Check if metadata already exists
      const existingMetadata = await storage.getMetadataByDatasetId(datasetId);
      
      let metadata;
      if (existingMetadata) {
        metadata = await storage.updateMetadata(existingMetadata.id, validatedData);
      } else {
        metadata = await storage.createMetadata(validatedData);
      }
      
      res.status(201).json({ metadata });
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ error: error.errors });
      }
      res.status(500).json({ error: "Failed to create/update metadata" });
    }
  });

  // Search datasets
  app.post("/api/search", async (req, res) => {
    try {
      const { query } = req.body;
      
      if (!query || typeof query !== "string") {
        return res.status(400).json({ error: "Query parameter is required" });
      }
      
      const searchResults = await semanticSearch.search(query, storage);
      
      // Store search query
      await storage.saveSearchQuery({
        query,
        processedQuery: searchResults.processedQuery,
        results: searchResults.results 
      });
      
      res.json(searchResults);
    } catch (error) {
      res.status(500).json({ error: "Search failed" });
    }
  });

  // Source datasets from external API
  app.post("/api/source-datasets", async (req, res) => {
    try {
      const { source, query, limit } = req.body;
      
      if (!source || !query) {
        return res.status(400).json({ error: "Source and query parameters are required" });
      }
      
      const datasets = await datasetService.sourceExternalDatasets(source, query, limit);
      res.json({ datasets });
    } catch (error) {
      res.status(500).json({ error: "Failed to source datasets" });
    }
  });

  // Generate metadata for a dataset
  app.post("/api/datasets/:id/generate-metadata", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      const dataset = await storage.getDataset(datasetId);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      const metadata = await metadataService.generateMetadata(dataset);
      
      // Save the generated metadata
      const existingMetadata = await storage.getMetadataByDatasetId(datasetId);
      
      let savedMetadata;
      if (existingMetadata) {
        savedMetadata = await storage.updateMetadata(existingMetadata.id, metadata);
      } else {
        savedMetadata = await storage.createMetadata({ ...metadata, datasetId });
      }
      
      res.json({ metadata: savedMetadata });
    } catch (error) {
      res.status(500).json({ error: "Failed to generate metadata" });
    }
  });

  // Download dataset
  app.get("/api/datasets/:id/download", async (req, res) => {
    try {
      const datasetId = parseInt(req.params.id);
      const dataset = await storage.getDataset(datasetId);
      
      if (!dataset) {
        return res.status(404).json({ error: "Dataset not found" });
      }
      
      // In a real implementation, this would trigger an actual download
      // For this MVP, we'll just return success
      res.json({ 
        success: true, 
        message: "Dataset download initiated", 
        downloadUrl: dataset.sourceUrl 
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to initiate download" });
    }
  });

  // Get stats about the datasets
  app.get("/api/stats", async (req, res) => {
    try {
      const datasets = await storage.getAllDatasets();
      
      const totalDatasets = datasets.length;
      const fairCompliant = datasets.filter(d => d.fairCompliant).length;
      const highQualityMetadata = datasets.filter(d => d.metadataQuality && d.metadataQuality >= 80).length;
      
      res.json({
        totalDatasets,
        fairCompliant,
        highQualityMetadata,
        apiCalls: Math.floor(Math.random() * 2000) + 1000 // Random number for demo
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch stats" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
