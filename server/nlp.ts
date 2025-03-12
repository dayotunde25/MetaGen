import natural from 'natural';
import { Dataset } from '@shared/schema';
import { storage } from './storage';

export class NlpManager {
  private tokenizer: any;
  private tfidf: any;
  private isInitialized: boolean;

  constructor() {
    // Use Tokenizer and TfIdf from natural
    this.tokenizer = new natural.WordTokenizer();
    this.tfidf = new natural.TfIdf();
    this.isInitialized = false;
  }

  // Initialize NLP components
  private async initialize() {
    if (this.isInitialized) return;
    
    // In a real application, we would train our models here
    // This is a simplified version for demonstration
    this.isInitialized = true;
  }

  // Tokenize and clean text
  private preprocessText(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2);
  }

  // Calculate TF-IDF score for terms
  private calculateTfIdf(term: string, document: string, allDocuments: string[]): number {
    const tf = document.split(/\s+/).filter(word => word === term).length / document.split(/\s+/).length;
    const documentsWithTerm = allDocuments.filter(doc => doc.includes(term)).length;
    const idf = Math.log(allDocuments.length / (documentsWithTerm || 1));
    return tf * idf;
  }

  // Extract keywords from text
  public async extractKeywords(text: string, count: number = 5): Promise<string[]> {
    await this.initialize();
    
    const tokens = this.preprocessText(text);
    const uniqueTokens = [...new Set(tokens)];
    
    // In a real implementation, we would use a more sophisticated algorithm
    // This is a simplified version that returns the most frequent tokens
    const tokenCounts = uniqueTokens.map(token => ({
      token,
      count: tokens.filter(t => t === token).length
    }));
    
    return tokenCounts
      .sort((a, b) => b.count - a.count)
      .slice(0, count)
      .map(item => item.token);
  }

  // Perform semantic search on datasets
  public async semanticSearch(query: string): Promise<Dataset[]> {
    await this.initialize();
    
    // Get all datasets
    const allDatasets = await storage.getDatasets();
    
    // Extract keywords from query
    const queryKeywords = await this.extractKeywords(query, 10);
    
    // Score datasets based on semantic relevance to query
    const scoredDatasets = allDatasets.map(dataset => {
      const datasetText = [
        dataset.name,
        dataset.description || '',
        (dataset.tags || []).join(' ')
      ].join(' ').toLowerCase();
      
      // Calculate relevance score based on keyword matching
      let score = 0;
      for (const keyword of queryKeywords) {
        if (datasetText.includes(keyword)) {
          score += 1;
          // Boost score for matches in title
          if (dataset.name.toLowerCase().includes(keyword)) {
            score += 2;
          }
          // Boost score for matches in tags
          if (dataset.tags && dataset.tags.some(tag => tag.toLowerCase().includes(keyword))) {
            score += 1.5;
          }
        }
      }
      
      return { dataset, score };
    });
    
    // Sort by relevance score and return datasets
    return scoredDatasets
      .sort((a, b) => b.score - a.score)
      .filter(item => item.score > 0)  // Only return relevant results
      .map(item => item.dataset);
  }

  // Suggest related tags based on dataset content
  public async suggestTags(text: string, count: number = 5): Promise<string[]> {
    await this.initialize();
    
    // Extract keywords that can serve as tags
    return this.extractKeywords(text, count);
  }

  // Classify dataset into categories
  public async classifyDataset(dataset: Dataset): Promise<string[]> {
    await this.initialize();
    
    const text = [
      dataset.name,
      dataset.description || '',
      (dataset.tags || []).join(' ')
    ].join(' ');
    
    // In a real implementation, we would use a trained classifier
    // This is a simplified version that looks for domain-specific keywords
    
    const categories = [];
    
    if (/machine learning|neural network|ai|deep learning|model/i.test(text)) {
      categories.push('Machine Learning');
    }
    
    if (/health|medical|patient|clinical|disease|hospital/i.test(text)) {
      categories.push('Healthcare');
    }
    
    if (/finance|market|stock|economic|bank|trading/i.test(text)) {
      categories.push('Financial');
    }
    
    if (/image|vision|recognition|detection|pixel/i.test(text)) {
      categories.push('Computer Vision');
    }
    
    if (/text|language|nlp|sentiment|translation/i.test(text)) {
      categories.push('NLP');
    }
    
    return categories;
  }
}
