import { datasets, type Dataset, type InsertDataset, processingQueue, type ProcessingQueue, type InsertProcessingQueue, metadataQuality, type MetadataQuality, type InsertMetadataQuality, users, type User, type InsertUser, type StatsData } from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Dataset operations
  getDatasets(): Promise<Dataset[]>;
  getDatasetById(id: number): Promise<Dataset | undefined>;
  getDatasetsByUser(userId: number): Promise<Dataset[]>;
  searchDatasets(query: string, tags?: string[], sortBy?: string, limit?: number, offset?: number): Promise<Dataset[]>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
  updateDataset(id: number, dataset: Partial<Dataset>): Promise<Dataset | undefined>;
  deleteDataset(id: number): Promise<boolean>;

  // Processing Queue operations
  getProcessingQueue(): Promise<ProcessingQueue[]>;
  getProcessingQueueByDatasetId(datasetId: number): Promise<ProcessingQueue | undefined>;
  addToProcessingQueue(queue: InsertProcessingQueue): Promise<ProcessingQueue>;
  updateProcessingQueue(id: number, queue: Partial<ProcessingQueue>): Promise<ProcessingQueue | undefined>;
  removeFromProcessingQueue(id: number): Promise<boolean>;

  // Metadata Quality operations
  getMetadataQuality(datasetId: number): Promise<MetadataQuality | undefined>;
  createMetadataQuality(quality: InsertMetadataQuality): Promise<MetadataQuality>;
  updateMetadataQuality(id: number, quality: Partial<MetadataQuality>): Promise<MetadataQuality | undefined>;

  // Stats
  getStats(): Promise<StatsData>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private datasets: Map<number, Dataset>;
  private processingQueue: Map<number, ProcessingQueue>;
  private metadataQuality: Map<number, MetadataQuality>;
  private currentUserId: number;
  private currentDatasetId: number;
  private currentQueueId: number;
  private currentQualityId: number;

  constructor() {
    this.users = new Map();
    this.datasets = new Map();
    this.processingQueue = new Map();
    this.metadataQuality = new Map();
    this.currentUserId = 1;
    this.currentDatasetId = 1;
    this.currentQueueId = 1;
    this.currentQualityId = 1;

    // Initialize with some demo data
    this.initializeDemoData();
  }

  private initializeDemoData() {
    // Add a demo user
    const demoUser: User = {
      id: this.currentUserId++,
      username: "demo",
      password: "demo123", // In a real application, this would be hashed
    };
    this.users.set(demoUser.id, demoUser);
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Dataset operations
  async getDatasets(): Promise<Dataset[]> {
    return Array.from(this.datasets.values());
  }

  async getDatasetById(id: number): Promise<Dataset | undefined> {
    return this.datasets.get(id);
  }

  async getDatasetsByUser(userId: number): Promise<Dataset[]> {
    return Array.from(this.datasets.values()).filter(
      (dataset) => dataset.userId === userId
    );
  }

  async searchDatasets(query: string, tags?: string[], sortBy: string = "dateAdded", limit: number = 10, offset: number = 0): Promise<Dataset[]> {
    let results = Array.from(this.datasets.values());
    
    // Filter by search query (check name and description)
    if (query) {
      const queryLower = query.toLowerCase();
      results = results.filter(
        dataset => 
          dataset.name.toLowerCase().includes(queryLower) || 
          (dataset.description && dataset.description.toLowerCase().includes(queryLower))
      );
    }
    
    // Filter by tags if provided
    if (tags && tags.length > 0) {
      results = results.filter(dataset => 
        dataset.tags && tags.some(tag => dataset.tags?.includes(tag))
      );
    }
    
    // Sort results
    switch (sortBy) {
      case "name":
        results.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case "quality":
        results.sort((a, b) => (b.qualityScore || 0) - (a.qualityScore || 0));
        break;
      case "dateAdded":
      default:
        results.sort((a, b) => {
          const dateA = a.dateAdded instanceof Date ? a.dateAdded : new Date(a.dateAdded);
          const dateB = b.dateAdded instanceof Date ? b.dateAdded : new Date(b.dateAdded);
          return dateB.getTime() - dateA.getTime();
        });
        break;
    }
    
    // Apply pagination
    return results.slice(offset, offset + limit);
  }

  async createDataset(insertDataset: InsertDataset): Promise<Dataset> {
    const id = this.currentDatasetId++;
    const now = new Date();
    
    const dataset: Dataset = {
      ...insertDataset,
      id,
      dateAdded: now,
      isProcessed: false,
      isFairCompliant: false,
      fairScore: 0,
      schemaOrgScore: 0,
      status: "queued",
      metadata: {}
    };
    
    this.datasets.set(id, dataset);
    return dataset;
  }

  async updateDataset(id: number, update: Partial<Dataset>): Promise<Dataset | undefined> {
    const dataset = this.datasets.get(id);
    if (!dataset) return undefined;
    
    const updatedDataset = { ...dataset, ...update };
    this.datasets.set(id, updatedDataset);
    return updatedDataset;
  }

  async deleteDataset(id: number): Promise<boolean> {
    return this.datasets.delete(id);
  }

  // Processing Queue operations
  async getProcessingQueue(): Promise<ProcessingQueue[]> {
    return Array.from(this.processingQueue.values());
  }

  async getProcessingQueueByDatasetId(datasetId: number): Promise<ProcessingQueue | undefined> {
    return Array.from(this.processingQueue.values()).find(
      (queue) => queue.datasetId === datasetId
    );
  }

  async addToProcessingQueue(insertQueue: InsertProcessingQueue): Promise<ProcessingQueue> {
    const id = this.currentQueueId++;
    
    const queue: ProcessingQueue = {
      ...insertQueue,
      id,
      progress: 0,
      startTime: null,
      endTime: null,
      estimatedCompletionTime: null,
      error: null
    };
    
    this.processingQueue.set(id, queue);
    return queue;
  }

  async updateProcessingQueue(id: number, update: Partial<ProcessingQueue>): Promise<ProcessingQueue | undefined> {
    const queue = this.processingQueue.get(id);
    if (!queue) return undefined;
    
    const updatedQueue = { ...queue, ...update };
    this.processingQueue.set(id, updatedQueue);
    return updatedQueue;
  }

  async removeFromProcessingQueue(id: number): Promise<boolean> {
    return this.processingQueue.delete(id);
  }

  // Metadata Quality operations
  async getMetadataQuality(datasetId: number): Promise<MetadataQuality | undefined> {
    return Array.from(this.metadataQuality.values()).find(
      (quality) => quality.datasetId === datasetId
    );
  }

  async createMetadataQuality(insertQuality: InsertMetadataQuality): Promise<MetadataQuality> {
    const id = this.currentQualityId++;
    
    const quality: MetadataQuality = {
      ...insertQuality,
      id
    };
    
    this.metadataQuality.set(id, quality);
    return quality;
  }

  async updateMetadataQuality(id: number, update: Partial<MetadataQuality>): Promise<MetadataQuality | undefined> {
    const quality = this.metadataQuality.get(id);
    if (!quality) return undefined;
    
    const updatedQuality = { ...quality, ...update };
    this.metadataQuality.set(id, updatedQuality);
    return updatedQuality;
  }

  // Stats
  async getStats(): Promise<StatsData> {
    const allDatasets = Array.from(this.datasets.values());
    
    return {
      totalDatasets: allDatasets.length,
      processedDatasets: allDatasets.filter(d => d.isProcessed).length,
      fairCompliantDatasets: allDatasets.filter(d => d.isFairCompliant).length,
      queuedDatasets: Array.from(this.processingQueue.values()).filter(q => q.status === "queued" || q.status === "processing").length
    };
  }
}

export const storage = new MemStorage();
