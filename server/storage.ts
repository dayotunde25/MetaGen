import { 
  User, InsertUser, 
  Dataset, InsertDataset,
  Metadata, InsertMetadata,
  ProcessingHistory, InsertProcessingHistory
} from "@shared/schema";

export interface IStorage {
  // User Operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Dataset Operations
  getAllDatasets(): Promise<Dataset[]>;
  getDataset(id: number): Promise<Dataset | undefined>;
  getDatasetsByStatus(status: string): Promise<Dataset[]>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
  updateDatasetStatus(id: number, status: string, progress?: number, etc?: string): Promise<Dataset | undefined>;
  getRecentDatasets(limit: number): Promise<Dataset[]>;
  searchDatasets(query: string, filters?: any): Promise<Dataset[]>;
  
  // Metadata Operations
  getMetadata(id: number): Promise<Metadata | undefined>;
  getMetadataByDatasetId(datasetId: number): Promise<Metadata | undefined>;
  createMetadata(metadata: InsertMetadata): Promise<Metadata>;
  
  // Processing History Operations
  getProcessingHistoryByDatasetId(datasetId: number): Promise<ProcessingHistory[]>;
  createProcessingHistory(history: InsertProcessingHistory): Promise<ProcessingHistory>;
  updateProcessingHistory(id: number, status: string, details?: string, endTime?: Date): Promise<ProcessingHistory | undefined>;
  getProcessingStatistics(): Promise<any>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private datasets: Map<number, Dataset>;
  private metadataRecords: Map<number, Metadata>;
  private processingHistories: Map<number, ProcessingHistory>;
  
  private userIdCounter: number;
  private datasetIdCounter: number;
  private metadataIdCounter: number;
  private processingHistoryIdCounter: number;

  constructor() {
    this.users = new Map();
    this.datasets = new Map();
    this.metadataRecords = new Map();
    this.processingHistories = new Map();
    
    this.userIdCounter = 1;
    this.datasetIdCounter = 1;
    this.metadataIdCounter = 1;
    this.processingHistoryIdCounter = 1;
    
    // Initialize with sample data
    this.initializeSampleData();
  }

  // User Operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userIdCounter++;
    const now = new Date();
    const user: User = { ...insertUser, id, createdAt: now };
    this.users.set(id, user);
    return user;
  }

  // Dataset Operations
  async getAllDatasets(): Promise<Dataset[]> {
    return Array.from(this.datasets.values());
  }

  async getDataset(id: number): Promise<Dataset | undefined> {
    return this.datasets.get(id);
  }

  async getDatasetsByStatus(status: string): Promise<Dataset[]> {
    return Array.from(this.datasets.values()).filter(
      (dataset) => dataset.status === status
    );
  }

  async createDataset(insertDataset: InsertDataset): Promise<Dataset> {
    const id = this.datasetIdCounter++;
    const now = new Date();
    const dataset: Dataset = {
      ...insertDataset,
      id,
      status: "pending",
      progress: 0,
      createdAt: now,
      updatedAt: now
    };
    
    this.datasets.set(id, dataset);
    return dataset;
  }

  async updateDatasetStatus(id: number, status: string, progress?: number, etc?: string): Promise<Dataset | undefined> {
    const dataset = this.datasets.get(id);
    if (!dataset) return undefined;
    
    const updatedDataset: Dataset = {
      ...dataset,
      status,
      progress: progress !== undefined ? progress : dataset.progress,
      estimatedTimeToCompletion: etc !== undefined ? etc : dataset.estimatedTimeToCompletion,
      updatedAt: new Date()
    };
    
    this.datasets.set(id, updatedDataset);
    return updatedDataset;
  }

  async getRecentDatasets(limit: number): Promise<Dataset[]> {
    return Array.from(this.datasets.values())
      .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
      .slice(0, limit);
  }

  async searchDatasets(query: string, filters?: any): Promise<Dataset[]> {
    let results = Array.from(this.datasets.values());
    
    if (query) {
      const lowerQuery = query.toLowerCase();
      results = results.filter(dataset => 
        dataset.title.toLowerCase().includes(lowerQuery) ||
        (dataset.description && dataset.description.toLowerCase().includes(lowerQuery)) ||
        dataset.source.toLowerCase().includes(lowerQuery) ||
        (dataset.keywords && dataset.keywords.some(keyword => keyword.toLowerCase().includes(lowerQuery)))
      );
    }
    
    if (filters) {
      if (filters.format) {
        results = results.filter(dataset => 
          dataset.formats && dataset.formats.includes(filters.format)
        );
      }
      
      if (filters.category) {
        results = results.filter(dataset => 
          dataset.category === filters.category
        );
      }
      
      if (filters.dateRange) {
        const now = new Date();
        let fromDate = new Date();
        
        switch (filters.dateRange) {
          case 'past-day':
            fromDate.setDate(now.getDate() - 1);
            break;
          case 'past-week':
            fromDate.setDate(now.getDate() - 7);
            break;
          case 'past-month':
            fromDate.setMonth(now.getMonth() - 1);
            break;
          case 'past-year':
            fromDate.setFullYear(now.getFullYear() - 1);
            break;
        }
        
        results = results.filter(dataset => 
          dataset.updatedAt >= fromDate
        );
      }
    }
    
    return results;
  }

  // Metadata Operations
  async getMetadata(id: number): Promise<Metadata | undefined> {
    return this.metadataRecords.get(id);
  }

  async getMetadataByDatasetId(datasetId: number): Promise<Metadata | undefined> {
    return Array.from(this.metadataRecords.values()).find(
      (metadata) => metadata.datasetId === datasetId
    );
  }

  async createMetadata(insertMetadata: InsertMetadata): Promise<Metadata> {
    const id = this.metadataIdCounter++;
    const now = new Date();
    const metadata: Metadata = {
      ...insertMetadata,
      id,
      createdAt: now
    };
    
    this.metadataRecords.set(id, metadata);
    return metadata;
  }

  // Processing History Operations
  async getProcessingHistoryByDatasetId(datasetId: number): Promise<ProcessingHistory[]> {
    return Array.from(this.processingHistories.values())
      .filter(history => history.datasetId === datasetId)
      .sort((a, b) => b.startTime.getTime() - a.startTime.getTime());
  }

  async createProcessingHistory(insertHistory: InsertProcessingHistory): Promise<ProcessingHistory> {
    const id = this.processingHistoryIdCounter++;
    const history: ProcessingHistory = {
      ...insertHistory,
      id
    };
    
    this.processingHistories.set(id, history);
    return history;
  }

  async updateProcessingHistory(id: number, status: string, details?: string, endTime?: Date): Promise<ProcessingHistory | undefined> {
    const history = this.processingHistories.get(id);
    if (!history) return undefined;
    
    const updatedHistory: ProcessingHistory = {
      ...history,
      status,
      details: details !== undefined ? details : history.details,
      endTime: endTime || new Date()
    };
    
    this.processingHistories.set(id, updatedHistory);
    return updatedHistory;
  }

  async getProcessingStatistics(): Promise<any> {
    const allDatasets = Array.from(this.datasets.values());
    const total = allDatasets.length;
    const processed = allDatasets.filter(d => d.status === 'processed').length;
    const processing = allDatasets.filter(d => ['downloading', 'structuring', 'generating'].includes(d.status)).length;
    const failed = allDatasets.filter(d => d.status === 'failed').length;
    
    // Calculate percentage changes (mock data for now)
    return {
      total,
      processed,
      processing,
      failed,
      totalChange: 12,
      processedChange: 8,
      failedChange: -30
    };
  }

  // Helper method to initialize sample data
  private initializeSampleData() {
    // Admin user
    const adminUser: InsertUser = {
      username: 'admin',
      password: 'password123', // In a real app, this would be hashed
      email: 'admin@research.org',
      fullName: 'Research Admin'
    };
    this.createUser(adminUser);
    
    // Sample datasets
    const covidDataset: InsertDataset = {
      title: 'COVID-19 Global Dataset',
      description: 'Comprehensive COVID-19 data including cases, deaths, and recoveries globally.',
      source: 'Johns Hopkins University',
      sourceUrl: 'https://github.com/CSSEGISandData/COVID-19',
      category: 'Healthcare',
      size: '2.3 GB',
      formats: ['CSV', 'JSON'],
      keywords: ['COVID-19', 'coronavirus', 'pandemic', 'epidemiology', 'public health', 'global']
    };
    
    const climateDataset: InsertDataset = {
      title: 'Global Surface Temperature',
      description: 'Long-term global surface temperature anomalies with baseline period 1951-1980.',
      source: 'NASA GISS',
      sourceUrl: 'https://data.giss.nasa.gov/gistemp/',
      category: 'Climate',
      size: '4.7 GB',
      formats: ['NetCDF', 'CSV'],
      keywords: ['climate', 'temperature', 'global warming', 'earth science']
    };
    
    const economicsDataset: InsertDataset = {
      title: 'World Development Indicators',
      description: 'Economic, social, and environmental indicators for countries worldwide since 1960.',
      source: 'World Bank',
      sourceUrl: 'https://datacatalog.worldbank.org/dataset/world-development-indicators',
      category: 'Economics',
      size: '1.8 GB',
      formats: ['Excel', 'JSON'],
      keywords: ['economics', 'development', 'global', 'indicators', 'social', 'environmental']
    };
    
    const satelliteDataset: InsertDataset = {
      title: 'Satellite Imagery Dataset',
      description: 'High-resolution satellite imagery for environmental monitoring.',
      source: 'NASA Earth Data',
      sourceUrl: 'https://earthdata.nasa.gov/',
      category: 'Earth Science',
      size: '2.3 GB',
      formats: ['GeoTIFF', 'HDF5'],
      keywords: ['satellite', 'imagery', 'remote sensing', 'earth observation']
    };
    
    const genomicDataset: InsertDataset = {
      title: 'Genomic Sequencing Data',
      description: 'Comprehensive genomic sequencing data for medical research.',
      source: 'NCBI',
      sourceUrl: 'https://www.ncbi.nlm.nih.gov/',
      category: 'Genomics',
      size: '6.7 GB',
      formats: ['FASTQ', 'BAM'],
      keywords: ['genomics', 'sequencing', 'biology', 'medicine']
    };
    
    const socialMediaDataset: InsertDataset = {
      title: 'Social Media Analytics',
      description: 'Dataset containing social media interactions and sentiment analysis.',
      source: 'Kaggle',
      sourceUrl: 'https://kaggle.com/datasets',
      category: 'Social Sciences',
      size: '1.2 GB',
      formats: ['CSV', 'JSON'],
      keywords: ['social media', 'analytics', 'sentiment', 'interactions']
    };
    
    // Create the datasets and set their status
    this.createDataset(covidDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'processed', 100);
      
      // Add metadata for this dataset
      const metadataInsert: InsertMetadata = {
        datasetId: dataset.id,
        schemaOrgJson: {
          "@context": "https://schema.org/",
          "@type": "Dataset",
          "name": "COVID-19 Global Dataset",
          "description": "This dataset contains global confirmed cases, recoveries, and deaths due to COVID-19, collected from various official sources.",
          "creator": {
            "@type": "Organization",
            "name": "Johns Hopkins University",
            "url": "https://www.jhu.edu/"
          },
          "publisher": {
            "@type": "Organization",
            "name": "Johns Hopkins CSSE",
            "url": "https://systems.jhu.edu/"
          },
          "datePublished": "2020-03-23",
          "dateModified": "2023-01-15",
          "license": "https://creativecommons.org/licenses/by/4.0/",
          "keywords": ["COVID-19", "coronavirus", "pandemic", "epidemiology", "public health", "global"]
        },
        fairAssessment: {
          findable: 95,
          accessible: 90,
          interoperable: 85,
          reusable: 92,
          overall: 91
        },
        dataStructure: [
          { field: "date", type: "Date (YYYY-MM-DD)", description: "Date of observation" },
          { field: "country_region", type: "String", description: "Country or region name" },
          { field: "province_state", type: "String", description: "Province or state (if applicable)" },
          { field: "lat", type: "Float", description: "Latitude coordinate" },
          { field: "long", type: "Float", description: "Longitude coordinate" }
        ],
        isFairCompliant: true,
        creator: "Johns Hopkins University",
        publisher: "Johns Hopkins CSSE",
        publicationDate: "2020-03-23",
        lastUpdated: "2023-01-15",
        language: "English",
        license: "CC BY 4.0",
        temporalCoverage: "2020-01-22/2023-01-15",
        spatialCoverage: "Global"
      };
      this.createMetadata(metadataInsert);
      
      // Add processing history
      const history: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'download',
        status: 'success',
        details: 'Downloaded from JHU GitHub repository',
        startTime: new Date(Date.now() - 1000000),
        endTime: new Date(Date.now() - 900000)
      };
      this.createProcessingHistory(history);
      
      const history2: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'structure',
        status: 'success',
        details: 'Structured according to schema.org standards',
        startTime: new Date(Date.now() - 800000),
        endTime: new Date(Date.now() - 700000)
      };
      this.createProcessingHistory(history2);
      
      const history3: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'metadata_generation',
        status: 'success',
        details: 'Generated comprehensive metadata',
        startTime: new Date(Date.now() - 600000),
        endTime: new Date(Date.now() - 500000)
      };
      this.createProcessingHistory(history3);
    });
    
    this.createDataset(climateDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'processing', 67, '~12 minutes');
    });
    
    this.createDataset(economicsDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'processed', 100);
      
      // Add metadata for this dataset
      const metadataInsert: InsertMetadata = {
        datasetId: dataset.id,
        schemaOrgJson: {
          "@context": "https://schema.org/",
          "@type": "Dataset",
          "name": "World Development Indicators",
          "description": "Economic, social, and environmental indicators for countries worldwide since 1960.",
          "creator": {
            "@type": "Organization",
            "name": "World Bank",
            "url": "https://www.worldbank.org/"
          },
          "publisher": {
            "@type": "Organization",
            "name": "World Bank Group",
            "url": "https://www.worldbank.org/"
          },
          "datePublished": "1960-01-01",
          "dateModified": "2023-01-15",
          "license": "https://creativecommons.org/licenses/by/4.0/",
          "keywords": ["economics", "development", "global", "indicators", "social", "environmental"]
        },
        fairAssessment: {
          findable: 90,
          accessible: 95,
          interoperable: 88,
          reusable: 92,
          overall: 91
        },
        dataStructure: [],
        isFairCompliant: true,
        creator: "World Bank",
        publisher: "World Bank Group",
        publicationDate: "1960-01-01",
        lastUpdated: "2023-01-15",
        language: "English",
        license: "CC BY 4.0",
        temporalCoverage: "1960-01-01/2023-01-15",
        spatialCoverage: "Global"
      };
      this.createMetadata(metadataInsert);
    });
    
    this.createDataset(satelliteDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'downloading', 45, '~35 minutes');
      
      // Add processing history
      const history: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'download',
        status: 'in_progress',
        details: 'Downloading from NASA Earth Data repository',
        startTime: new Date(Date.now() - 500000)
      };
      this.createProcessingHistory(history);
    });
    
    this.createDataset(genomicDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'structuring', 78, '~12 minutes');
      
      // Add processing history
      const history: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'download',
        status: 'success',
        details: 'Downloaded from NCBI repository',
        startTime: new Date(Date.now() - 400000),
        endTime: new Date(Date.now() - 300000)
      };
      this.createProcessingHistory(history);
      
      const history2: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'structure',
        status: 'in_progress',
        details: 'Structuring genomic data according to schema.org standards',
        startTime: new Date(Date.now() - 200000)
      };
      this.createProcessingHistory(history2);
    });
    
    this.createDataset(socialMediaDataset).then(dataset => {
      this.updateDatasetStatus(dataset.id, 'generating', 92, '~3 minutes');
      
      // Add processing history
      const history: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'download',
        status: 'success',
        details: 'Downloaded from Kaggle',
        startTime: new Date(Date.now() - 300000),
        endTime: new Date(Date.now() - 250000)
      };
      this.createProcessingHistory(history);
      
      const history2: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'structure',
        status: 'success',
        details: 'Structured according to schema.org standards',
        startTime: new Date(Date.now() - 200000),
        endTime: new Date(Date.now() - 150000)
      };
      this.createProcessingHistory(history2);
      
      const history3: InsertProcessingHistory = {
        datasetId: dataset.id,
        operation: 'metadata_generation',
        status: 'in_progress',
        details: 'Generating metadata',
        startTime: new Date(Date.now() - 100000)
      };
      this.createProcessingHistory(history3);
    });
  }
}

export const storage = new MemStorage();
