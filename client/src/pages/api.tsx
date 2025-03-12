import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { CodeIcon, KeyIcon, SearchIcon, DatabaseIcon, ServerIcon, InfoIcon } from "lucide-react";
import { CopyToClipboard } from "@/components/ui/copy-to-clipboard";

const ApiPage = () => {
  const [apiKey, setApiKey] = useState<string | null>(null);

  // In a real app, this would fetch the API key from the server
  const generateApiKey = () => {
    setApiKey("sk_datameta_" + Math.random().toString(36).substring(2, 15));
  };

  return (
    <main className="flex-grow bg-gray-50">
      <div className="bg-primary-700 text-white">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-2">
            <CodeIcon className="h-6 w-6" />
            <h1 className="text-2xl font-extrabold sm:text-3xl">API Reference</h1>
          </div>
          <p className="mt-2 text-lg max-w-3xl">
            Integrate DataMeta AI into your applications with our powerful REST API
          </p>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-4">
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <Card>
                <CardHeader>
                  <CardTitle>API Access</CardTitle>
                  <CardDescription>
                    Get your API key to use DataMeta AI in your applications
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {apiKey ? (
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm font-medium mb-1">Your API Key:</p>
                        <div className="relative">
                          <div className="bg-gray-100 p-2 rounded font-mono text-sm break-all pr-9">
                            {apiKey}
                          </div>
                          <CopyToClipboard
                            text={apiKey}
                            className="absolute right-2 top-2"
                          />
                        </div>
                        <p className="text-xs text-gray-500 mt-2">
                          Keep this key secure. Do not share it in public repositories or client-side code.
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full"
                        onClick={generateApiKey}
                      >
                        <KeyIcon className="h-4 w-4 mr-2" />
                        Regenerate API Key
                      </Button>
                    </div>
                  ) : (
                    <Button
                      className="w-full"
                      onClick={generateApiKey}
                    >
                      <KeyIcon className="h-4 w-4 mr-2" />
                      Generate API Key
                    </Button>
                  )}
                </CardContent>
              </Card>
              
              <div className="mt-6 space-y-2">
                <div className="text-sm font-medium">Quick Navigation</div>
                <ul className="space-y-1">
                  <li>
                    <a href="#authentication" className="text-sm text-primary-600 hover:text-primary-700">Authentication</a>
                  </li>
                  <li>
                    <a href="#rate-limits" className="text-sm text-primary-600 hover:text-primary-700">Rate Limits</a>
                  </li>
                  <li>
                    <a href="#datasets" className="text-sm text-primary-600 hover:text-primary-700">Datasets</a>
                  </li>
                  <li>
                    <a href="#metadata" className="text-sm text-primary-600 hover:text-primary-700">Metadata</a>
                  </li>
                  <li>
                    <a href="#search" className="text-sm text-primary-600 hover:text-primary-700">Search</a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="lg:col-span-3">
            <Card>
              <CardHeader>
                <CardTitle>Getting Started</CardTitle>
                <CardDescription>
                  Introduction to the DataMeta AI API
                </CardDescription>
              </CardHeader>
              <CardContent className="prose max-w-none">
                <p>
                  The DataMeta AI API allows you to source, process, and generate structured metadata for AI research datasets programmatically. This API follows RESTful principles and uses JSON for all requests and responses.
                </p>
                
                <h3 id="authentication" className="flex items-center">
                  <KeyIcon className="h-5 w-5 mr-2 text-primary-500" />
                  Authentication
                </h3>
                <Separator className="my-3" />
                <p>
                  All API requests must include your API key in the Authorization header:
                </p>
                <div className="bg-gray-800 rounded-md text-white p-4 font-mono text-sm">
                  <CopyToClipboard
                    text={`Authorization: Bearer ${apiKey || 'YOUR_API_KEY'}`}
                    className="absolute right-6 top-4 text-white"
                  />
                  <code>Authorization: Bearer {apiKey || 'YOUR_API_KEY'}</code>
                </div>
                
                <h3 id="rate-limits" className="flex items-center mt-8">
                  <InfoIcon className="h-5 w-5 mr-2 text-primary-500" />
                  Rate Limits
                </h3>
                <Separator className="my-3" />
                <p>
                  The API is rate limited to protect our infrastructure. Current limits are:
                </p>
                <ul>
                  <li>100 requests per minute</li>
                  <li>5,000 requests per day</li>
                </ul>
                <p>
                  If you exceed these limits, the API will return a <code>429 Too Many Requests</code> response.
                </p>
                
                <h3 id="datasets" className="flex items-center mt-8">
                  <DatabaseIcon className="h-5 w-5 mr-2 text-primary-500" />
                  Datasets Endpoints
                </h3>
                <Separator className="my-3" />
                
                <Tabs defaultValue="list" className="w-full">
                  <TabsList>
                    <TabsTrigger value="list">List Datasets</TabsTrigger>
                    <TabsTrigger value="get">Get Dataset</TabsTrigger>
                    <TabsTrigger value="create">Create Dataset</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="list" className="mt-4">
                    <h4 className="text-lg font-semibold">List All Datasets</h4>
                    <p className="text-sm text-gray-600 mb-2">GET /api/datasets</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}'
  }
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "datasets": [
    {
      "id": 1,
      "title": "Climate Change: Global Temperature Time Series",
      "description": "Comprehensive dataset of global temperature measurements spanning 150 years.",
      "source": "National Oceanic and Atmospheric Administration (NOAA)",
      "sourceUrl": "https://data.noaa.gov/dataset/global-temperature-time-series",
      "dataType": "Time Series",
      "category": "Climate",
      "size": "2.7 GB",
      "format": "CSV, JSON, NetCDF",
      "recordCount": 15768945,
      "fairCompliant": true,
      "metadataQuality": 88,
      "createdAt": "2023-07-15T10:30:00Z",
      "updatedAt": "2023-07-15T10:30:00Z"
    },
    // Additional datasets...
  ]
}`}</code></pre>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="get" className="mt-4">
                    <h4 className="text-lg font-semibold">Get a Specific Dataset</h4>
                    <p className="text-sm text-gray-600 mb-2">GET /api/datasets/{'{id}'}</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets/1', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}'
  }
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "dataset": {
    "id": 1,
    "title": "Climate Change: Global Temperature Time Series",
    "description": "Comprehensive dataset of global temperature measurements spanning 150 years.",
    "source": "National Oceanic and Atmospheric Administration (NOAA)",
    "sourceUrl": "https://data.noaa.gov/dataset/global-temperature-time-series",
    "dataType": "Time Series",
    "category": "Climate",
    "size": "2.7 GB",
    "format": "CSV, JSON, NetCDF",
    "recordCount": 15768945,
    "fairCompliant": true,
    "metadataQuality": 88,
    "createdAt": "2023-07-15T10:30:00Z",
    "updatedAt": "2023-07-15T10:30:00Z"
  }
}`}</code></pre>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="create" className="mt-4">
                    <h4 className="text-lg font-semibold">Create a New Dataset</h4>
                    <p className="text-sm text-gray-600 mb-2">POST /api/datasets</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "title": "Financial Market Analysis Dataset",
    "description": "Historical stock market data for predictive modeling.",
    "source": "NYSE",
    "sourceUrl": "https://example.com/nyse/datasets",
    "dataType": "Time Series",
    "category": "Financial",
    "size": "1.5 GB",
    "format": "CSV, JSON",
    "recordCount": 8750000,
    "fairCompliant": true,
    "metadataQuality": 75
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "dataset": {
    "id": 3,
    "title": "Financial Market Analysis Dataset",
    "description": "Historical stock market data for predictive modeling.",
    "source": "NYSE",
    "sourceUrl": "https://example.com/nyse/datasets",
    "dataType": "Time Series",
    "category": "Financial",
    "size": "1.5 GB",
    "format": "CSV, JSON",
    "recordCount": 8750000,
    "fairCompliant": true,
    "metadataQuality": 75,
    "createdAt": "2023-10-05T14:22:30Z",
    "updatedAt": "2023-10-05T14:22:30Z"
  }
}`}</code></pre>
                    </div>
                  </TabsContent>
                </Tabs>
                
                <h3 id="metadata" className="flex items-center mt-8">
                  <ServerIcon className="h-5 w-5 mr-2 text-primary-500" />
                  Metadata Endpoints
                </h3>
                <Separator className="my-3" />
                
                <Tabs defaultValue="get-metadata" className="w-full">
                  <TabsList>
                    <TabsTrigger value="get-metadata">Get Metadata</TabsTrigger>
                    <TabsTrigger value="create-metadata">Create Metadata</TabsTrigger>
                    <TabsTrigger value="generate">Generate Metadata</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="get-metadata" className="mt-4">
                    <h4 className="text-lg font-semibold">Get Metadata for a Dataset</h4>
                    <p className="text-sm text-gray-600 mb-2">GET /api/datasets/{'{id}'}/metadata</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets/1/metadata', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}'
  }
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "metadata": {
    "id": 1,
    "datasetId": 1,
    "schemaOrgJson": {
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
    "fairScores": {
      "findable": 95,
      "accessible": 85,
      "interoperable": 90,
      "reusable": 80
    },
    "keywords": ["climate change", "global warming", "temperature", "time series", "meteorology"],
    "variableMeasured": ["Average global temperature", "Land temperature", "Ocean temperature", "Temperature anomaly"],
    "temporalCoverage": "1850-01-01/2023-06-30",
    "spatialCoverage": {
      "@type": "Place",
      "geo": {
        "@type": "GeoShape",
        "box": "-90 -180 90 180"
      }
    },
    "license": "CC BY 4.0",
    "createdAt": "2023-07-15T10:30:00Z",
    "updatedAt": "2023-07-15T10:30:00Z"
  }
}`}</code></pre>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="create-metadata" className="mt-4">
                    <h4 className="text-lg font-semibold">Create or Update Metadata</h4>
                    <p className="text-sm text-gray-600 mb-2">POST /api/datasets/{'{id}'}/metadata</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets/3/metadata', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "schemaOrgJson": {
      "@context": "https://schema.org/",
      "@type": "Dataset",
      "name": "Financial Market Analysis Dataset",
      "description": "Historical stock market data for predictive modeling.",
      "url": "https://example.com/nyse/datasets",
      "keywords": ["finance", "stocks", "market", "trading", "nyse"],
      "creator": {
        "@type": "Organization",
        "name": "NYSE",
        "url": "https://www.nyse.com/"
      },
      "datePublished": "2023-10-05",
      "license": "https://creativecommons.org/licenses/by/4.0/",
      "variableMeasured": [
        "Stock price",
        "Trading volume",
        "Market cap",
        "Volatility"
      ],
      "temporalCoverage": "2010-01-01/2023-09-30"
    },
    "fairScores": {
      "findable": 85,
      "accessible": 90,
      "interoperable": 75,
      "reusable": 80
    },
    "keywords": ["finance", "stocks", "market", "trading", "nyse"],
    "variableMeasured": ["Stock price", "Trading volume", "Market cap", "Volatility"],
    "temporalCoverage": "2010-01-01/2023-09-30",
    "license": "CC BY 4.0"
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "metadata": {
    "id": 3,
    "datasetId": 3,
    "schemaOrgJson": { /* schema.org JSON as above */ },
    "fairScores": {
      "findable": 85,
      "accessible": 90,
      "interoperable": 75,
      "reusable": 80
    },
    "keywords": ["finance", "stocks", "market", "trading", "nyse"],
    "variableMeasured": ["Stock price", "Trading volume", "Market cap", "Volatility"],
    "temporalCoverage": "2010-01-01/2023-09-30",
    "spatialCoverage": null,
    "license": "CC BY 4.0",
    "createdAt": "2023-10-05T14:25:30Z",
    "updatedAt": "2023-10-05T14:25:30Z"
  }
}`}</code></pre>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="generate" className="mt-4">
                    <h4 className="text-lg font-semibold">Generate Metadata Automatically</h4>
                    <p className="text-sm text-gray-600 mb-2">POST /api/datasets/{'{id}'}/generate-metadata</p>
                    
                    <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                      <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/datasets/3/generate-metadata', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}',
    'Content-Type': 'application/json'
  }
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "metadata": {
    "id": 3,
    "datasetId": 3,
    "schemaOrgJson": {
      "@context": "https://schema.org/",
      "@type": "Dataset",
      // Automatically generated schema.org JSON...
    },
    "fairScores": {
      "findable": 88,
      "accessible": 92,
      "interoperable": 78,
      "reusable": 85
    },
    // Additional metadata fields...
    "createdAt": "2023-10-05T14:30:15Z",
    "updatedAt": "2023-10-05T14:30:15Z"
  }
}`}</code></pre>
                    </div>
                  </TabsContent>
                </Tabs>
                
                <h3 id="search" className="flex items-center mt-8">
                  <SearchIcon className="h-5 w-5 mr-2 text-primary-500" />
                  Search Endpoints
                </h3>
                <Separator className="my-3" />
                
                <h4 className="text-lg font-semibold">Semantic Search for Datasets</h4>
                <p className="text-sm text-gray-600 mb-2">POST /api/search</p>
                
                <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                  <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/search', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "query": "climate change temperature global warming"
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "query": "climate change temperature global warming",
  "processedQuery": "climate change temperature global warming",
  "results": [
    {
      "dataset": {
        "id": 1,
        "title": "Climate Change: Global Temperature Time Series",
        "description": "Comprehensive dataset of global temperature measurements spanning 150 years.",
        // Additional dataset fields...
      },
      "score": 9.8
    },
    // Additional search results...
  ],
  "timing": {
    "total": 235,
    "processing": 87,
    "search": 148
  }
}`}</code></pre>
                </div>
                
                <h4 className="text-lg font-semibold mt-6">Source External Datasets</h4>
                <p className="text-sm text-gray-600 mb-2">POST /api/source-datasets</p>
                
                <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto font-mono text-xs">
                  <pre><code>{`// Example Request
fetch('https://api.datameta.ai/api/source-datasets', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey || 'YOUR_API_KEY'}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "source": "kaggle",
    "query": "healthcare patient records",
    "limit": 5
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Example Response
{
  "datasets": [
    {
      "title": "Healthcare Dataset: healthcare patient records Analysis #1",
      "description": "A comprehensive dataset for healthcare patient records research and analysis.",
      "source": "Kaggle",
      "sourceUrl": "https://example.com/kaggle/datasets/healthcare-patient-records-1",
      "dataType": "Tabular",
      "category": "Healthcare",
      "size": "45 MB",
      "format": "CSV, JSON",
      "recordCount": 56432
    },
    // Additional datasets...
  ]
}`}</code></pre>
                </div>
              </CardContent>
            </Card>
            
            <div className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle>SDKs and Libraries</CardTitle>
                  <CardDescription>
                    Use our official client libraries to integrate with DataMeta AI
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="border rounded-md p-4 text-center">
                      <div className="font-bold text-lg mb-2">JavaScript</div>
                      <p className="text-sm text-gray-500 mb-4">For browser and Node.js applications</p>
                      <Button variant="outline" size="sm">
                        <CodeIcon className="h-4 w-4 mr-2" />
                        View on npm
                      </Button>
                    </div>
                    
                    <div className="border rounded-md p-4 text-center">
                      <div className="font-bold text-lg mb-2">Python</div>
                      <p className="text-sm text-gray-500 mb-4">For data science and ML applications</p>
                      <Button variant="outline" size="sm">
                        <CodeIcon className="h-4 w-4 mr-2" />
                        View on PyPI
                      </Button>
                    </div>
                    
                    <div className="border rounded-md p-4 text-center">
                      <div className="font-bold text-lg mb-2">R</div>
                      <p className="text-sm text-gray-500 mb-4">For statistical computing and graphics</p>
                      <Button variant="outline" size="sm">
                        <CodeIcon className="h-4 w-4 mr-2" />
                        View on CRAN
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

// Using the imported CopyToClipboard component from "@/components/ui/copy-to-clipboard"

export default ApiPage;
