import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { BookOpenIcon, CodeIcon, DatabaseIcon, ClipboardListIcon, GitMergeIcon } from "lucide-react";

const Documentation = () => {
  return (
    <main className="flex-grow bg-gray-50">
      <div className="bg-primary-700 text-white">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-2">
            <BookOpenIcon className="h-6 w-6" />
            <h1 className="text-2xl font-extrabold sm:text-3xl">Documentation</h1>
          </div>
          <p className="mt-2 text-lg max-w-3xl">
            Learn how to use DataMeta AI to enhance your research datasets with structured metadata
          </p>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <Tabs defaultValue="getting-started" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="getting-started">Getting Started</TabsTrigger>
            <TabsTrigger value="metadata">Metadata Standards</TabsTrigger>
            <TabsTrigger value="fair">FAIR Principles</TabsTrigger>
            <TabsTrigger value="api">API Reference</TabsTrigger>
          </TabsList>
          
          <TabsContent value="getting-started">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Getting Started with DataMeta AI</CardTitle>
                <CardDescription>
                  Learn how to source, download, and enhance datasets with high-quality metadata
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                <div>
                  <h3 className="text-xl font-semibold flex items-center">
                    <DatabaseIcon className="mr-2 h-5 w-5 text-primary-500" />
                    Sourcing Datasets
                  </h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    DataMeta AI can automatically source datasets from various repositories including:
                  </p>
                  <ul className="list-disc pl-6 space-y-2 text-gray-700">
                    <li>Kaggle</li>
                    <li>Data.World</li>
                    <li>Figshare</li>
                    <li>Zenodo</li>
                    <li>Harvard Dataverse</li>
                    <li>Google Dataset Search</li>
                  </ul>
                  <p className="mt-4 text-gray-700">
                    To search for datasets, use the search bar on the dashboard or go to the Datasets page and click "Source New Dataset".
                  </p>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold flex items-center">
                    <ClipboardListIcon className="mr-2 h-5 w-5 text-primary-500" />
                    Processing Datasets
                  </h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    Once you've found a dataset, DataMeta AI will:
                  </p>
                  <ol className="list-decimal pl-6 space-y-2 text-gray-700">
                    <li>Download the dataset from its source</li>
                    <li>Analyze the dataset structure and content</li>
                    <li>Assess the dataset's quality</li>
                    <li>Generate structured metadata</li>
                    <li>Evaluate FAIR compliance</li>
                  </ol>
                  <p className="mt-4 text-gray-700">
                    This process may take a few minutes depending on the size and complexity of the dataset.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold flex items-center">
                    <CodeIcon className="mr-2 h-5 w-5 text-primary-500" />
                    Using Generated Metadata
                  </h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    After processing, you can:
                  </p>
                  <ul className="list-disc pl-6 space-y-2 text-gray-700">
                    <li>View the metadata in the dataset details</li>
                    <li>Download the metadata as a JSON-LD file</li>
                    <li>Include the metadata in your research publications</li>
                    <li>Embed the schema.org markup in your research websites</li>
                    <li>Use the metadata to enhance dataset discoverability</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="metadata">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Metadata Standards</CardTitle>
                <CardDescription>
                  Understanding the metadata standards used by DataMeta AI
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                <div>
                  <h3 className="text-xl font-semibold">Schema.org Dataset Standard</h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    DataMeta AI uses the Schema.org Dataset vocabulary to structure metadata. This standard is widely recognized and enables better discoverability of datasets through search engines.
                  </p>
                  <p className="text-gray-700 mb-4">
                    The Schema.org Dataset standard includes properties such as:
                  </p>
                  <ul className="list-disc pl-6 space-y-2 text-gray-700">
                    <li><span className="font-mono text-sm">name</span> - The name of the dataset</li>
                    <li><span className="font-mono text-sm">description</span> - A description of the dataset</li>
                    <li><span className="font-mono text-sm">url</span> - URL of the dataset</li>
                    <li><span className="font-mono text-sm">sameAs</span> - URL of a reference web page that identifies the dataset</li>
                    <li><span className="font-mono text-sm">keywords</span> - Keywords or tags used to describe the dataset</li>
                    <li><span className="font-mono text-sm">creator</span> - The creator/author of this dataset</li>
                    <li><span className="font-mono text-sm">datePublished</span> - Date of first publication</li>
                    <li><span className="font-mono text-sm">license</span> - A license document that applies to this dataset</li>
                    <li><span className="font-mono text-sm">variableMeasured</span> - The variable(s) measured in this dataset</li>
                    <li><span className="font-mono text-sm">temporalCoverage</span> - The time period that the dataset covers</li>
                    <li><span className="font-mono text-sm">spatialCoverage</span> - The spatial coverage of the dataset</li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold">JSON-LD Format</h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    DataMeta AI generates metadata in JSON-LD format (JavaScript Object Notation for Linked Data), which is the recommended format for implementing Schema.org.
                  </p>
                  <div className="bg-gray-800 rounded-md text-white p-4 overflow-auto max-h-60 font-mono text-xs">
                    <pre><code>{`{
  "@context": "https://schema.org/",
  "@type": "Dataset",
  "name": "Climate Change: Global Temperature Time Series",
  "description": "Comprehensive dataset of global temperature measurements spanning 150 years.",
  "url": "https://data.noaa.gov/dataset/global-temperature-time-series",
  "sameAs": "https://doi.org/10.7289/V5KD1VF2",
  "keywords": ["climate change", "global warming", "temperature"],
  "creator": {
    "@type": "Organization",
    "name": "National Oceanic and Atmospheric Administration",
    "url": "https://www.noaa.gov/"
  },
  "datePublished": "2023-07-15",
  "license": "https://creativecommons.org/licenses/by/4.0/"
}`}</code></pre>
                  </div>
                  <p className="mt-4 text-gray-700">
                    This format can be embedded in HTML pages using script tags or served via API endpoints.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="fair">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">FAIR Data Principles</CardTitle>
                <CardDescription>
                  Ensuring your datasets are Findable, Accessible, Interoperable, and Reusable
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                <div>
                  <h3 className="text-xl font-semibold flex items-center">
                    <GitMergeIcon className="mr-2 h-5 w-5 text-primary-500" />
                    Understanding FAIR Principles
                  </h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    The FAIR Data Principles are a set of guiding principles to make data Findable, Accessible, Interoperable, and Reusable. DataMeta AI evaluates and enhances datasets according to these principles.
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-bold text-blue-800 mb-2">Findable</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                        <li>Data should be easy to find by both humans and computer systems</li>
                        <li>Datasets should have rich metadata</li>
                        <li>Data should be assigned a globally unique and persistent identifier</li>
                        <li>Metadata should clearly include the identifier of the data it describes</li>
                      </ul>
                    </div>
                    
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-bold text-green-800 mb-2">Accessible</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                        <li>Data should be retrievable using standardized protocols</li>
                        <li>These protocols should be open, free, and universally implementable</li>
                        <li>The protocol should allow for authentication and authorization where necessary</li>
                        <li>Metadata should be accessible even when the data is no longer available</li>
                      </ul>
                    </div>
                    
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-bold text-purple-800 mb-2">Interoperable</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                        <li>Data should use a formal, accessible, shared, and broadly applicable language</li>
                        <li>Data should use vocabularies that follow FAIR principles</li>
                        <li>Data should include qualified references to other data</li>
                        <li>Metadata should use standards that promote data integration</li>
                      </ul>
                    </div>
                    
                    <div className="bg-yellow-50 p-4 rounded-lg">
                      <h4 className="font-bold text-yellow-800 mb-2">Reusable</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                        <li>Data should have a plurality of accurate and relevant attributes</li>
                        <li>Data should be released with a clear and accessible data usage license</li>
                        <li>Data should be associated with detailed provenance</li>
                        <li>Data should meet domain-relevant community standards</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold">FAIR Assessment in DataMeta AI</h3>
                  <Separator className="my-3" />
                  <p className="text-gray-700 mb-4">
                    DataMeta AI evaluates datasets against FAIR principles and provides a score for each component:
                  </p>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Principle</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">What We Check</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score Impact</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Findable</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Title, description, keywords, DOIs</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0-100%</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Accessible</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">URLs, open formats, access protocols</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0-100%</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Interoperable</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Standard formats, schema.org compliance</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0-100%</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Reusable</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">License information, provenance, quality</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0-100%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="api">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">API Quick Reference</CardTitle>
                <CardDescription>
                  Overview of the DataMeta AI API endpoints
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 mb-6">
                  Visit the <a href="/api" className="text-primary-600 hover:underline">full API documentation</a> for detailed information, request/response examples, and authentication details.
                </p>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Endpoint</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">GET</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Get all datasets</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">GET</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Get a specific dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">POST</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Create a new dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">PATCH</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Update a dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">DELETE</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Delete a dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}/metadata</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">GET</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Get metadata for a dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}/metadata</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">POST</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Create or update metadata</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/search</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">POST</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Search datasets with NLP</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/source-datasets</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">POST</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Source datasets from external repositories</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}/generate-metadata</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">POST</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Generate metadata for a dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/datasets/{'{id}'}/download</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">GET</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Download a dataset</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary-600">/api/stats</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">GET</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Get dataset statistics</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
};

export default Documentation;
