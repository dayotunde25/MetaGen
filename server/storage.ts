import { 
  users, type User, type InsertUser, 
  datasets, type Dataset, type InsertDataset,
  metadata, type Metadata, type InsertMetadata,
  searchQueries, type SearchQuery, type InsertSearchQuery
} from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Dataset operations
  getAllDatasets(): Promise<Dataset[]>;
  getDataset(id: number): Promise<Dataset | undefined>;
  getDatasetsBySource(source: string): Promise<Dataset[]>;
  getDatasetsByCategory(category: string): Promise<Dataset[]>;
  getDatasetsByType(dataType: string): Promise<Dataset[]>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
  updateDataset(id: number, dataset: Partial<Dataset>): Promise<Dataset | undefined>;
  deleteDataset(id: number): Promise<boolean>;

  // Metadata operations
  getMetadata(id: number): Promise<Metadata | undefined>;
  getMetadataByDatasetId(datasetId: number): Promise<Metadata | undefined>;
  createMetadata(metadata: InsertMetadata): Promise<Metadata>;
  updateMetadata(id: number, metadata: Partial<Metadata>): Promise<Metadata | undefined>;

  // Search operations
  saveSearchQuery(query: InsertSearchQuery): Promise<SearchQuery>;
  getRecentSearchQueries(limit: number): Promise<SearchQuery[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private datasets: Map<number, Dataset>;
  private metadataItems: Map<number, Metadata>;
  private searches: Map<number, SearchQuery>;
  
  private userCurrentId: number;
  private datasetCurrentId: number;
  private metadataCurrentId: number;
  private searchCurrentId: number;

  constructor() {
    this.users = new Map();
    this.datasets = new Map();
    this.metadataItems = new Map();
    this.searches = new Map();
    
    this.userCurrentId = 1;
    this.datasetCurrentId = 1;
    this.metadataCurrentId = 1;
    this.searchCurrentId = 1;

    // Initialize with some sample datasets
    this.seedInitialData();
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userCurrentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Dataset operations
  async getAllDatasets(): Promise<Dataset[]> {
    return Array.from(this.datasets.values());
  }

  async getDataset(id: number): Promise<Dataset | undefined> {
    return this.datasets.get(id);
  }

  async getDatasetsBySource(source: string): Promise<Dataset[]> {
    return Array.from(this.datasets.values())
      .filter(dataset => dataset.source.toLowerCase().includes(source.toLowerCase()));
  }

  async getDatasetsByCategory(category: string): Promise<Dataset[]> {
    return Array.from(this.datasets.values())
      .filter(dataset => dataset.category.toLowerCase() === category.toLowerCase());
  }

  async getDatasetsByType(dataType: string): Promise<Dataset[]> {
    return Array.from(this.datasets.values())
      .filter(dataset => dataset.dataType.toLowerCase() === dataType.toLowerCase());
  }

  async createDataset(insertDataset: InsertDataset): Promise<Dataset> {
    const id = this.datasetCurrentId++;
    const now = new Date();
    const dataset: Dataset = { 
      id,
      title: insertDataset.title,
      description: insertDataset.description,
      source: insertDataset.source,
      sourceUrl: insertDataset.sourceUrl,
      dataType: insertDataset.dataType,
      category: insertDataset.category,
      size: insertDataset.size || null,
      format: insertDataset.format || null,
      recordCount: insertDataset.recordCount || null,
      fairCompliant: insertDataset.fairCompliant || null,
      metadataQuality: insertDataset.metadataQuality || null,
      createdAt: now,
      updatedAt: now
    };
    this.datasets.set(id, dataset);
    return dataset;
  }

  async updateDataset(id: number, datasetUpdate: Partial<Dataset>): Promise<Dataset | undefined> {
    const existingDataset = this.datasets.get(id);
    if (!existingDataset) return undefined;

    const updatedDataset: Dataset = {
      ...existingDataset,
      ...datasetUpdate,
      updatedAt: new Date()
    };
    this.datasets.set(id, updatedDataset);
    return updatedDataset;
  }

  async deleteDataset(id: number): Promise<boolean> {
    return this.datasets.delete(id);
  }

  // Metadata operations
  async getMetadata(id: number): Promise<Metadata | undefined> {
    return this.metadataItems.get(id);
  }

  async getMetadataByDatasetId(datasetId: number): Promise<Metadata | undefined> {
    return Array.from(this.metadataItems.values())
      .find(metadata => metadata.datasetId === datasetId);
  }

  async createMetadata(insertMetadata: InsertMetadata): Promise<Metadata> {
    const id = this.metadataCurrentId++;
    const now = new Date();
    const metadata: Metadata = {
      id,
      datasetId: insertMetadata.datasetId,
      schemaOrgJson: insertMetadata.schemaOrgJson,
      fairScores: insertMetadata.fairScores,
      keywords: insertMetadata.keywords || null,
      variableMeasured: insertMetadata.variableMeasured || null,
      temporalCoverage: insertMetadata.temporalCoverage || null,
      spatialCoverage: insertMetadata.spatialCoverage || null,
      license: insertMetadata.license || null,
      createdAt: now,
      updatedAt: now
    };
    this.metadataItems.set(id, metadata);
    return metadata;
  }

  async updateMetadata(id: number, metadataUpdate: Partial<Metadata>): Promise<Metadata | undefined> {
    const existingMetadata = this.metadataItems.get(id);
    if (!existingMetadata) return undefined;

    const updatedMetadata: Metadata = {
      ...existingMetadata,
      ...metadataUpdate,
      updatedAt: new Date()
    };
    this.metadataItems.set(id, updatedMetadata);
    return updatedMetadata;
  }

  // Search operations
  async saveSearchQuery(insertSearchQuery: InsertSearchQuery): Promise<SearchQuery> {
    const id = this.searchCurrentId++;
    const now = new Date();
    const searchQuery: SearchQuery = {
      ...insertSearchQuery,
      id,
      createdAt: now
    };
    this.searches.set(id, searchQuery);
    return searchQuery;
  }

  async getRecentSearchQueries(limit: number): Promise<SearchQuery[]> {
    return Array.from(this.searches.values())
      .sort((a, b) => {
        if (!a.createdAt || !b.createdAt) return 0;
        return b.createdAt.getTime() - a.createdAt.getTime();
      })
      .slice(0, limit);
  }

  // Seed initial data
  private seedInitialData() {
    // Climate Change dataset
    const climateDataset: InsertDataset = {
      title: "Climate Change: Global Temperature Time Series",
      description: "Comprehensive dataset of global temperature measurements spanning 150 years.",
      source: "National Oceanic and Atmospheric Administration (NOAA)",
      sourceUrl: "https://data.noaa.gov/dataset/global-temperature-time-series",
      dataType: "Time Series",
      category: "Climate",
      size: "2.7 GB",
      format: "CSV, JSON, NetCDF",
      recordCount: 15768945,
      fairCompliant: true,
      metadataQuality: 88
    };
    
    const climateDatasetEntity = this.createDataset(climateDataset);
    
    // Voice dataset
    const voiceDataset: InsertDataset = {
      title: "Common Voice Speech Corpus",
      description: "Multi-language speech dataset with 13,905 hours of validated recordings.",
      source: "Mozilla Foundation",
      sourceUrl: "https://commonvoice.mozilla.org/datasets",
      dataType: "Audio",
      category: "NLP",
      size: "76.4 GB",
      format: "MP3, JSON, TSV",
      recordCount: 13905,
      fairCompliant: true,
      metadataQuality: 92
    };
    
    const voiceDatasetEntity = this.createDataset(voiceDataset);

    // Add metadata for climate dataset
    climateDatasetEntity.then(dataset => {
      this.createMetadata({
        datasetId: dataset.id,
        schemaOrgJson: {
          "@context": "https://schema.org/",
          "@type": "Dataset",
          "name": "Climate Change: Global Temperature Time Series",
          "description": "Comprehensive dataset of global temperature measurements spanning 150 years.",
          "url": "https://data.noaa.gov/dataset/global-temperature-time-series",
          "sameAs": "https://doi.org/10.7289/V5KD1VF2",
          "keywords": [
            "climate change",
            "global warming",
            "temperature",
            "time series",
            "meteorology"
          ],
          "creator": {
            "@type": "Organization",
            "name": "National Oceanic and Atmospheric Administration",
            "url": "https://www.noaa.gov/"
          },
          "datePublished": "2023-07-15",
          "license": "https://creativecommons.org/licenses/by/4.0/",
          "variableMeasured": [
            "Average global temperature",
            "Land temperature",
            "Ocean temperature",
            "Temperature anomaly"
          ],
          "temporalCoverage": "1850-01-01/2023-06-30",
          "spatialCoverage": {
            "@type": "Place",
            "geo": {
              "@type": "GeoShape",
              "box": "-90 -180 90 180"
            }
          }
        },
        fairScores: {
          findable: 95,
          accessible: 85,
          interoperable: 90,
          reusable: 80
        },
        keywords: ["climate change", "global warming", "temperature", "time series", "meteorology"],
        variableMeasured: ["Average global temperature", "Land temperature", "Ocean temperature", "Temperature anomaly"],
        temporalCoverage: "1850-01-01/2023-06-30",
        spatialCoverage: {
          "@type": "Place",
          "geo": {
            "@type": "GeoShape",
            "box": "-90 -180 90 180"
          }
        },
        license: "CC BY 4.0"
      });
    });

    // Add metadata for voice dataset
    voiceDatasetEntity.then(dataset => {
      this.createMetadata({
        datasetId: dataset.id,
        schemaOrgJson: {
          "@context": "https://schema.org/",
          "@type": "Dataset",
          "name": "Common Voice Speech Corpus",
          "description": "Multi-language speech dataset with 13,905 hours of validated recordings.",
          "url": "https://commonvoice.mozilla.org/datasets",
          "sameAs": "https://doi.org/10.5281/zenodo.2499178",
          "keywords": [
            "speech",
            "audio",
            "voice",
            "language",
            "NLP",
            "corpus"
          ],
          "creator": {
            "@type": "Organization",
            "name": "Mozilla Foundation",
            "url": "https://foundation.mozilla.org/"
          },
          "datePublished": "2023-08-03",
          "license": "https://creativecommons.org/licenses/by/4.0/",
          "variableMeasured": [
            "Speech audio",
            "Speaker demographics",
            "Speech transcriptions",
            "Language"
          ],
          "temporalCoverage": "2017-01-01/2023-07-30"
        },
        fairScores: {
          findable: 92,
          accessible: 95,
          interoperable: 88,
          reusable: 93
        },
        keywords: ["speech", "audio", "voice", "language", "NLP", "corpus"],
        variableMeasured: ["Speech audio", "Speaker demographics", "Speech transcriptions", "Language"],
        temporalCoverage: "2017-01-01/2023-07-30",
        license: "CC BY 4.0"
      });
    });
  }
}

export const storage = new MemStorage();
