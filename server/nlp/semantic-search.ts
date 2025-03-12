import { IStorage } from "../storage";
import { Dataset } from "@shared/schema";
import { NlpManager } from "node-nlp";

interface SearchResult {
  dataset: Dataset;
  score: number;
}

export interface SearchResponse {
  query: string;
  processedQuery: string;
  results: SearchResult[];
  timing: {
    total: number;
    processing: number;
    search: number;
  };
}

export class SemanticSearch {
  private nlp: NlpManager;

  constructor() {
    this.nlp = new NlpManager({ languages: ['en'] });
  }

  async search(query: string, storage: IStorage): Promise<SearchResponse> {
    const startTime = Date.now();
    
    // Process the query with NLP
    const processingStart = Date.now();
    const processedQuery = await this.processQuery(query);
    const processingTime = Date.now() - processingStart;
    
    // Fetch all datasets
    const searchStart = Date.now();
    const datasets = await storage.getAllDatasets();
    
    // Search using processed keywords
    const results = await this.searchDatasets(processedQuery, datasets);
    const searchTime = Date.now() - searchStart;
    
    // Calculate total time
    const totalTime = Date.now() - startTime;
    
    return {
      query,
      processedQuery: processedQuery.join(' '),
      results,
      timing: {
        total: totalTime,
        processing: processingTime,
        search: searchTime
      }
    };
  }

  private async processQuery(query: string): Promise<string[]> {
    // Extract keywords from the query using simple tokenization
    const tokens = query.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2);
    
    // Remove common stop words
    const stopWords = ['and', 'the', 'with', 'for', 'that', 'this', 'are', 'from'];
    const keywords = tokens.filter(word => !stopWords.includes(word));
    
    // Return unique keywords using filter instead of Set
    return keywords.filter((word, index) => keywords.indexOf(word) === index);
  }

  private async searchDatasets(keywords: string[], datasets: Dataset[]): Promise<SearchResult[]> {
    const results: SearchResult[] = [];
    
    for (const dataset of datasets) {
      // Calculate relevance score based on keyword matches
      const score = this.calculateRelevanceScore(keywords, dataset);
      
      if (score > 0) {
        results.push({
          dataset,
          score
        });
      }
    }
    
    // Sort by relevance score in descending order
    return results.sort((a, b) => b.score - a.score);
  }

  private calculateRelevanceScore(keywords: string[], dataset: Dataset): number {
    let score = 0;
    
    // Search in various fields with different weights
    const searchFields = [
      { field: dataset.title, weight: 3 },
      { field: dataset.description, weight: 2 },
      { field: dataset.source, weight: 1 },
      { field: dataset.category, weight: 2 },
      { field: dataset.dataType, weight: 2 },
      { field: dataset.format || '', weight: 1 }
    ];
    
    for (const keyword of keywords) {
      const keywordLower = keyword.toLowerCase();
      
      for (const { field, weight } of searchFields) {
        if (field.toLowerCase().includes(keywordLower)) {
          score += weight;
        }
      }
    }
    
    // Boost score for high quality metadata and FAIR compliance
    if (dataset.metadataQuality && dataset.metadataQuality > 80) {
      score *= 1.2;
    }
    
    if (dataset.fairCompliant) {
      score *= 1.1;
    }
    
    return score;
  }
}
