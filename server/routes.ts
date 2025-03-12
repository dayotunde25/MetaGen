import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { NlpManager } from "./nlp";
import { DatasetFetcher } from "./dataset-fetcher";
import { DatasetProcessor } from "./dataset-processor";
import { MetadataGenerator } from "./metadata-generator";
import { insertDatasetSchema, type InsertDataset, insertProcessingQueueSchema } from "@shared/schema";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";

export async function registerRoutes(app: Express): Promise<Server> {
  // Initialize services
  const nlpManager = new NlpManager();
  const datasetFetcher = new DatasetFetcher();
  const datasetProcessor = new DatasetProcessor();
  const metadataGenerator = new MetadataGenerator();

  // Get stats
  app.get("/api/stats", async (_req: Request, res: Response) => {
    try {
      const stats = await storage.getStats();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ message: `Error fetching stats: ${error.message}` });
    }
  });

  // Dataset CRUD operations
  app.get("/api/datasets", async (_req: Request, res: Response) => {
    try {
      const datasets = await storage.getDatasets();
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ message: `Error fetching datasets: ${error.message}` });
    }
  });

  app.get("/api/datasets/:id", async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const dataset = await storage.getDatasetById(id);
      
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      // Get metadata quality if available
      const quality = await storage.getMetadataQuality(id);
      
      // Get processing status if in queue
      const processing = await storage.getProcessingQueueByDatasetId(id);
      
      res.json({
        ...dataset,
        quality,
        processing
      });
    } catch (error) {
      res.status(500).json({ message: `Error fetching dataset: ${error.message}` });
    }
  });

  app.post("/api/datasets", async (req: Request, res: Response) => {
    try {
      const validatedData = insertDatasetSchema.parse(req.body);
      const dataset = await storage.createDataset(validatedData);
      
      // Add to processing queue
      const queueItem = await storage.addToProcessingQueue({
        datasetId: dataset.id,
        status: "queued"
      });
      
      res.status(201).json({ ...dataset, processingQueueId: queueItem.id });
    } catch (error) {
      if (error instanceof ZodError) {
        const validationError = fromZodError(error);
        return res.status(400).json({ message: validationError.message });
      }
      res.status(500).json({ message: `Error creating dataset: ${error.message}` });
    }
  });

  app.put("/api/datasets/:id", async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const dataset = await storage.updateDataset(id, req.body);
      
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      res.json(dataset);
    } catch (error) {
      res.status(500).json({ message: `Error updating dataset: ${error.message}` });
    }
  });

  app.delete("/api/datasets/:id", async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const success = await storage.deleteDataset(id);
      
      if (!success) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      res.status(204).send();
    } catch (error) {
      res.status(500).json({ message: `Error deleting dataset: ${error.message}` });
    }
  });

  // Search datasets
  app.get("/api/search", async (req: Request, res: Response) => {
    try {
      const query = req.query.q as string || "";
      const tags = req.query.tags ? (req.query.tags as string).split(",") : [];
      const sortBy = req.query.sortBy as string || "dateAdded";
      const limit = parseInt(req.query.limit as string || "10");
      const offset = parseInt(req.query.offset as string || "0");
      
      // If using semantic search and query is not empty
      if (query && req.query.semantic === "true") {
        const semanticResults = await nlpManager.semanticSearch(query);
        res.json(semanticResults);
      } else {
        // Regular search
        const results = await storage.searchDatasets(query, tags, sortBy, limit, offset);
        res.json(results);
      }
    } catch (error) {
      res.status(500).json({ message: `Error searching datasets: ${error.message}` });
    }
  });

  // Processing queue operations
  app.get("/api/processing-queue", async (_req: Request, res: Response) => {
    try {
      const queue = await storage.getProcessingQueue();
      res.json(queue);
    } catch (error) {
      res.status(500).json({ message: `Error fetching processing queue: ${error.message}` });
    }
  });

  app.post("/api/process-dataset/:id", async (req: Request, res: Response) => {
    try {
      const datasetId = parseInt(req.params.id);
      const dataset = await storage.getDatasetById(datasetId);
      
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      // Check if dataset already in queue
      const existingQueue = await storage.getProcessingQueueByDatasetId(datasetId);
      if (existingQueue && (existingQueue.status === "queued" || existingQueue.status === "processing")) {
        return res.status(409).json({ 
          message: "Dataset already in processing queue",
          queueItem: existingQueue
        });
      }
      
      // Add to queue
      const queueItem = await storage.addToProcessingQueue({
        datasetId,
        status: "queued"
      });
      
      // Update dataset status
      await storage.updateDataset(datasetId, { status: "queued" });
      
      // Start processing in background
      processDatasetInBackground(datasetId, queueItem.id);
      
      res.status(202).json({
        message: "Dataset added to processing queue",
        queueItem
      });
    } catch (error) {
      res.status(500).json({ message: `Error adding dataset to processing queue: ${error.message}` });
    }
  });

  // Metadata quality endpoints
  app.get("/api/metadata-quality/:datasetId", async (req: Request, res: Response) => {
    try {
      const datasetId = parseInt(req.params.datasetId);
      const quality = await storage.getMetadataQuality(datasetId);
      
      if (!quality) {
        return res.status(404).json({ message: "Metadata quality not found for this dataset" });
      }
      
      res.json(quality);
    } catch (error) {
      res.status(500).json({ message: `Error fetching metadata quality: ${error.message}` });
    }
  });

  // Background processing function
  async function processDatasetInBackground(datasetId: number, queueId: number) {
    try {
      // Update queue status to processing
      await storage.updateProcessingQueue(queueId, {
        status: "processing",
        startTime: new Date(),
        estimatedCompletionTime: new Date(Date.now() + 15 * 60 * 1000) // Estimate 15 min
      });
      
      // Update dataset status
      await storage.updateDataset(datasetId, { status: "processing" });
      
      // Fake progress updates (In a real implementation this would be actual progress)
      const updateInterval = setInterval(async () => {
        const currentQueue = await storage.getProcessingQueueByDatasetId(datasetId);
        if (!currentQueue || currentQueue.status !== "processing") {
          clearInterval(updateInterval);
          return;
        }
        
        const newProgress = Math.min((currentQueue.progress || 0) + 10, 95);
        await storage.updateProcessingQueue(queueId, { progress: newProgress });
      }, 5000);
      
      // 1. Fetch the dataset
      const dataset = await storage.getDatasetById(datasetId);
      if (!dataset) {
        throw new Error("Dataset not found");
      }
      
      // 2. Download dataset (simulation)
      const downloadedData = await datasetFetcher.fetchDataset(dataset.url);
      await storage.updateProcessingQueue(queueId, { progress: 30 });
      
      // 3. Process the dataset
      const processedData = await datasetProcessor.processDataset(downloadedData, dataset.format);
      await storage.updateProcessingQueue(queueId, { progress: 60 });
      
      // 4. Generate metadata
      const { metadata, quality } = await metadataGenerator.generateMetadata(processedData, dataset);
      await storage.updateProcessingQueue(queueId, { progress: 90 });
      
      // 5. Save metadata quality metrics
      await storage.createMetadataQuality({
        datasetId,
        fairFindable: quality.fairFindable,
        fairAccessible: quality.fairAccessible,
        fairInteroperable: quality.fairInteroperable,
        fairReusable: quality.fairReusable,
        schemaOrgRequired: quality.schemaOrgRequired,
        schemaOrgRecommended: quality.schemaOrgRecommended,
        schemaOrgVocabulary: quality.schemaOrgVocabulary,
        schemaOrgStructure: quality.schemaOrgStructure
      });
      
      // 6. Update dataset with metadata and scores
      const fairScore = Math.round(
        (quality.fairFindable + quality.fairAccessible + quality.fairInteroperable + quality.fairReusable) / 4
      );
      
      const schemaOrgScore = Math.round(
        (quality.schemaOrgRequired + quality.schemaOrgRecommended + quality.schemaOrgVocabulary + quality.schemaOrgStructure) / 4
      );
      
      const qualityScore = Math.round((fairScore + schemaOrgScore) / 2);
      
      await storage.updateDataset(datasetId, {
        metadata,
        fairScore,
        schemaOrgScore,
        qualityScore,
        isProcessed: true,
        isFairCompliant: fairScore >= 70, // Consider FAIR compliant if score >= 70
        status: "processed"
      });
      
      // 7. Mark processing as complete
      clearInterval(updateInterval);
      await storage.updateProcessingQueue(queueId, {
        status: "completed",
        progress: 100,
        endTime: new Date()
      });
      
    } catch (error) {
      console.error(`Error processing dataset ${datasetId}:`, error);
      
      // Update queue and dataset with error status
      await storage.updateProcessingQueue(queueId, {
        status: "error",
        error: error.message,
        endTime: new Date()
      });
      
      await storage.updateDataset(datasetId, { status: "error" });
    }
  }

  const httpServer = createServer(app);
  return httpServer;
}
