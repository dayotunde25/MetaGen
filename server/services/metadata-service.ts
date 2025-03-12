import { Dataset, InsertMetadata } from "@shared/schema";
import { IStorage } from "../storage";

interface FAIRScores {
  findable: number;
  accessible: number;
  interoperable: number;
  reusable: number;
}

export class MetadataService {
  private storage: IStorage;

  constructor(storage: IStorage) {
    this.storage = storage;
  }

  async generateMetadata(dataset: Dataset): Promise<Omit<InsertMetadata, "datasetId">> {
    // In a real implementation, this would analyze the dataset and generate proper metadata
    // For this MVP, we'll simulate the metadata generation process

    // Generate keywords based on title and description
    const keywords = this.extractKeywords(dataset.title + " " + dataset.description);

    // Generate variables measured based on dataset type and category
    const variableMeasured = this.generateVariableMeasured(dataset.dataType, dataset.category);

    // Generate temporal coverage
    const temporalCoverage = this.generateTemporalCoverage();

    // Generate spatial coverage if applicable
    const spatialCoverage = this.generateSpatialCoverage(dataset.category);

    // Generate license
    const license = "CC BY 4.0";

    // Calculate FAIR scores
    const fairScores = this.calculateFAIRScores(dataset, keywords, variableMeasured);

    // Generate schema.org JSON
    const schemaOrgJson = this.generateSchemaOrgJson(
      dataset,
      keywords,
      variableMeasured,
      temporalCoverage,
      spatialCoverage,
      license
    );

    return {
      schemaOrgJson,
      fairScores,
      keywords,
      variableMeasured,
      temporalCoverage,
      spatialCoverage,
      license
    };
  }

  private extractKeywords(text: string): string[] {
    // Simple keyword extraction
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter(word => word.length > 3);

    // Remove duplicates and limit to 10 keywords
    return [...new Set(words)].slice(0, 10);
  }

  private generateVariableMeasured(dataType: string, category: string): string[] {
    // Simplified variable generation based on data type and category
    const variables: Record<string, Record<string, string[]>> = {
      "Time Series": {
        "Climate": ["Average temperature", "Precipitation", "Humidity", "Wind speed"],
        "Financial": ["Stock price", "Trading volume", "Market cap", "Volatility"],
        "default": ["Value", "Timestamp", "Measurement", "Reading"]
      },
      "Tabular": {
        "Healthcare": ["Patient data", "Treatment outcomes", "Diagnostic measures", "Vital signs"],
        "Financial": ["Revenue", "Expenses", "Profit", "Growth rate"],
        "default": ["ID", "Name", "Value", "Category"]
      },
      "Image": {
        "Computer Vision": ["Object boundaries", "Classification labels", "Segmentation masks", "Feature vectors"],
        "default": ["Resolution", "Color channels", "Dimensions", "Format"]
      },
      "Text": {
        "NLP": ["Word frequency", "Sentiment score", "Named entities", "Part of speech tags"],
        "default": ["Word count", "Character count", "Language", "Encoding"]
      },
      "Audio": {
        "default": ["Duration", "Sample rate", "Transcription", "Speaker metadata"]
      },
      "default": {
        "default": ["Value", "Type", "Count", "Measurement"]
      }
    };

    // Get variables for the specific data type and category, or use defaults
    const typeVars = variables[dataType] || variables["default"];
    const vars = typeVars[category] || typeVars["default"];

    return vars;
  }

  private generateTemporalCoverage(): string {
    // Generate a random time range covering the last 10 years
    const endYear = new Date().getFullYear();
    const startYear = endYear - Math.floor(Math.random() * 10) - 1;
    return `${startYear}-01-01/${endYear}-12-31`;
  }

  private generateSpatialCoverage(category: string): any {
    // Only generate spatial coverage for certain categories
    if (["Climate", "Geographic", "Environmental"].includes(category)) {
      return {
        "@type": "Place",
        "geo": {
          "@type": "GeoShape",
          "box": "-90 -180 90 180" // Global coverage
        }
      };
    }
    return null;
  }

  private calculateFAIRScores(
    dataset: Dataset,
    keywords: string[],
    variableMeasured: string[]
  ): FAIRScores {
    // Calculate simulated FAIR scores based on dataset properties
    let findable = 50;
    let accessible = 50;
    let interoperable = 50;
    let reusable = 50;

    // Findable score factors
    if (dataset.title.length > 10) findable += 10;
    if (dataset.description.length > 50) findable += 10;
    if (keywords.length >= 5) findable += 10;
    if (dataset.sourceUrl.includes("doi.org")) findable += 20;

    // Accessible score factors
    if (dataset.sourceUrl) accessible += 20;
    if (dataset.format && dataset.format.includes("CSV")) accessible += 10;
    if (dataset.format && dataset.format.includes("JSON")) accessible += 10;
    if (dataset.fairCompliant) accessible += 10;

    // Interoperable score factors
    if (dataset.format && ["CSV", "JSON", "XML"].some(f => dataset.format!.includes(f))) interoperable += 20;
    if (variableMeasured.length >= 3) interoperable += 20;
    if (dataset.metadataQuality > 70) interoperable += 10;

    // Reusable score factors
    if (dataset.source) reusable += 15;
    if (dataset.fairCompliant) reusable += 15;
    if (dataset.metadataQuality > 80) reusable += 10;
    if (dataset.format && dataset.format.includes("CSV")) reusable += 10;

    // Cap scores at 100
    return {
      findable: Math.min(findable, 100),
      accessible: Math.min(accessible, 100),
      interoperable: Math.min(interoperable, 100),
      reusable: Math.min(reusable, 100)
    };
  }

  private generateSchemaOrgJson(
    dataset: Dataset,
    keywords: string[],
    variableMeasured: string[],
    temporalCoverage: string,
    spatialCoverage: any,
    license: string
  ): any {
    // Generate Schema.org compliant JSON-LD
    const schemaOrgJson: any = {
      "@context": "https://schema.org/",
      "@type": "Dataset",
      "name": dataset.title,
      "description": dataset.description,
      "url": dataset.sourceUrl,
      "keywords": keywords,
      "creator": {
        "@type": "Organization",
        "name": dataset.source,
        "url": dataset.sourceUrl
      },
      "datePublished": new Date().toISOString().split('T')[0],
      "license": license === "CC BY 4.0" ? "https://creativecommons.org/licenses/by/4.0/" : license,
      "variableMeasured": variableMeasured
    };

    if (temporalCoverage) {
      schemaOrgJson.temporalCoverage = temporalCoverage;
    }

    if (spatialCoverage) {
      schemaOrgJson.spatialCoverage = spatialCoverage;
    }

    return schemaOrgJson;
  }
}
