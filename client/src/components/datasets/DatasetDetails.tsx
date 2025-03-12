import { Dataset } from "@shared/schema";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tab";
import { 
  X, Download, Code, Eye, CheckCircle, Info, AlertCircle 
} from "lucide-react";
import { 
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow 
} from "@/components/ui/table";

interface DatasetDetailsProps {
  dataset: Dataset;
  onClose: () => void;
}

export default function DatasetDetails({ dataset, onClose }: DatasetDetailsProps) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="flex justify-between items-center p-4 border-b border-slate-200">
        <h2 className="text-xl font-semibold text-slate-900">{dataset.name}</h2>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-500">
          <X className="h-5 w-5" />
        </button>
      </div>
      
      <div className="p-4">
        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/3">
            <div className="rounded-lg overflow-hidden bg-slate-100 h-48">
              {dataset.thumbnail && (
                <img 
                  src={dataset.thumbnail} 
                  className="w-full h-full object-cover" 
                  alt={`${dataset.name} thumbnail`} 
                />
              )}
            </div>
            <div className="mt-4 bg-slate-50 rounded-lg p-4">
              <h3 className="font-medium text-slate-800 mb-2">Dataset Overview</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Source</span>
                  <span className="text-sm font-medium">{dataset.source}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Format</span>
                  <span className="text-sm font-medium">{dataset.format}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Size</span>
                  <span className="text-sm font-medium">{dataset.size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Records</span>
                  <span className="text-sm font-medium">{dataset.records}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Last Updated</span>
                  <span className="text-sm font-medium">{dataset.lastUpdated}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">License</span>
                  <span className="text-sm font-medium">{dataset.license}</span>
                </div>
              </div>
            </div>
            <div className="mt-4 flex flex-col space-y-2">
              <Button className="w-full py-2 bg-primary-600 hover:bg-primary-700 text-white">
                <Download className="h-4 w-4 mr-2" />
                <span>Download Dataset</span>
              </Button>
              <Button variant="outline" className="w-full py-2 border border-slate-300 text-slate-700 hover:bg-slate-50">
                <Code className="h-4 w-4 mr-2" />
                <span>API Access</span>
              </Button>
            </div>
          </div>
          
          <div className="md:w-2/3">
            <div className="border-b border-slate-200">
              <nav className="flex -mb-px">
                <button 
                  onClick={() => setActiveTab("overview")}
                  className={`py-3 px-4 border-b-2 font-medium text-sm ${
                    activeTab === "overview" 
                      ? "border-primary-500 text-primary-600" 
                      : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                  }`}
                >
                  Overview
                </button>
                <button 
                  onClick={() => setActiveTab("metadata")}
                  className={`py-3 px-4 border-b-2 font-medium text-sm ${
                    activeTab === "metadata" 
                      ? "border-primary-500 text-primary-600" 
                      : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                  }`}
                >
                  Metadata
                </button>
                <button 
                  onClick={() => setActiveTab("structure")}
                  className={`py-3 px-4 border-b-2 font-medium text-sm ${
                    activeTab === "structure" 
                      ? "border-primary-500 text-primary-600" 
                      : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                  }`}
                >
                  Structure
                </button>
                <button 
                  onClick={() => setActiveTab("quality")}
                  className={`py-3 px-4 border-b-2 font-medium text-sm ${
                    activeTab === "quality" 
                      ? "border-primary-500 text-primary-600" 
                      : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                  }`}
                >
                  Quality Assessment
                </button>
              </nav>
            </div>
            
            <div className="py-4">
              {/* Overview Tab */}
              {activeTab === "overview" && (
                <div>
                  <h3 className="font-semibold text-lg mb-3">About this Dataset</h3>
                  <p className="text-slate-600 mb-4">{dataset.description}</p>
                  
                  <h4 className="font-medium text-slate-800 mb-2">Citation</h4>
                  <div className="bg-slate-50 p-3 rounded font-mono text-sm mb-4">
                    {dataset.citation || `${dataset.source}. (${new Date().getFullYear()}). ${dataset.name}.`}
                  </div>
                  
                  <h4 className="font-medium text-slate-800 mb-2">Tags</h4>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {dataset.tags.map((tag, index) => (
                      <Badge
                        key={index}
                        className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-slate-100 text-slate-800"
                      >
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  
                  <h4 className="font-medium text-slate-800 mb-2">FAIR Assessment</h4>
                  <div className="mb-4">
                    <div className="flex items-center mb-2">
                      <div className="mr-3 font-medium">Overall Score:</div>
                      <div className="flex items-center">
                        <div className="w-24 h-3 bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-green-500 rounded-full" 
                            style={{ width: `${dataset.fairScore}%` }}
                          ></div>
                        </div>
                        <div className="ml-3 font-semibold">{dataset.fairScore}%</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm font-medium mb-1">Findability</div>
                        <div className="flex items-center">
                          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary-500 rounded-full" 
                              style={{ width: '95%' }}
                            ></div>
                          </div>
                          <div className="ml-2 text-sm font-medium">95%</div>
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-medium mb-1">Accessibility</div>
                        <div className="flex items-center">
                          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary-500 rounded-full" 
                              style={{ width: '90%' }}
                            ></div>
                          </div>
                          <div className="ml-2 text-sm font-medium">90%</div>
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-medium mb-1">Interoperability</div>
                        <div className="flex items-center">
                          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary-500 rounded-full" 
                              style={{ width: '85%' }}
                            ></div>
                          </div>
                          <div className="ml-2 text-sm font-medium">85%</div>
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-medium mb-1">Reusability</div>
                        <div className="flex items-center">
                          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary-500 rounded-full" 
                              style={{ width: '88%' }}
                            ></div>
                          </div>
                          <div className="ml-2 text-sm font-medium">88%</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Metadata Tab */}
              {activeTab === "metadata" && (
                <div>
                  <div className="flex justify-between mb-4">
                    <h3 className="font-semibold text-lg">Structured Metadata</h3>
                    <div>
                      <button className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center">
                        <Download className="h-4 w-4 mr-1" />
                        <span>Export JSON-LD</span>
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-slate-50 p-4 rounded-lg mb-4 font-mono text-sm overflow-auto max-h-96">
                    <pre className="whitespace-pre-wrap text-slate-700">
                      {JSON.stringify({
                        "@context": "https://schema.org/",
                        "@type": "Dataset",
                        "name": dataset.name,
                        "description": dataset.description,
                        "url": `https://example.org/datasets/${dataset.id}`,
                        "sameAs": `https://doi.org/10.xxxx/xxxxx`,
                        "version": "2.1.0",
                        "license": `https://creativecommons.org/licenses/${dataset.license.includes('CC BY') ? 'by/4.0/' : 'cc0/1.0/'}`,
                        "creator": {
                          "@type": "Organization",
                          "name": dataset.source,
                          "url": `https://www.example.org/`
                        },
                        "temporalCoverage": "1970-01-01/2020-12-31",
                        "spatialCoverage": {
                          "@type": "Place",
                          "geo": {
                            "@type": "GeoShape",
                            "box": "-90 -180 90 180"
                          }
                        },
                        "variableMeasured": [
                          "variable_1",
                          "variable_2",
                          "variable_3"
                        ],
                        "distribution": {
                          "@type": "DataDownload",
                          "contentUrl": `https://example.org/datasets/${dataset.id}.${dataset.format.toLowerCase()}`,
                          "encodingFormat": dataset.format,
                          "contentSize": dataset.size
                        },
                        "keywords": dataset.tags,
                        "datePublished": dataset.lastUpdated,
                        "dateModified": dataset.lastUpdated
                      }, null, 2)}
                    </pre>
                  </div>
                  
                  <h4 className="font-medium text-slate-800 mb-2">Schema.org Compliance</h4>
                  <div className="flex items-center mb-4">
                    <div className="w-24 h-3 bg-slate-200 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 rounded-full" style={{ width: '98%' }}></div>
                    </div>
                    <div className="ml-3 text-sm">
                      <span className="font-semibold">98% compliant</span>
                      <span className="text-slate-500"> with schema.org/Dataset</span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Structure Tab */}
              {activeTab === "structure" && (
                <div>
                  <h3 className="font-semibold text-lg mb-3">Dataset Structure</h3>
                  
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-slate-800">Field Definitions</h4>
                      <span className="text-sm text-slate-500">5 columns, {dataset.records || 'Unknown'} rows</span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                      <Table>
                        <TableHeader className="bg-slate-50">
                          <TableRow>
                            <TableHead>Field Name</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Description</TableHead>
                            <TableHead>Example</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableRow>
                            <TableCell className="font-medium text-slate-900">field_1</TableCell>
                            <TableCell>string</TableCell>
                            <TableCell>First sample field</TableCell>
                            <TableCell>"value"</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium text-slate-900">field_2</TableCell>
                            <TableCell>float</TableCell>
                            <TableCell>Second sample field</TableCell>
                            <TableCell>4.2</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium text-slate-900">field_3</TableCell>
                            <TableCell>float</TableCell>
                            <TableCell>Third sample field</TableCell>
                            <TableCell>856.7</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium text-slate-900">field_4</TableCell>
                            <TableCell>float</TableCell>
                            <TableCell>Fourth sample field</TableCell>
                            <TableCell>15.3</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium text-slate-900">field_5</TableCell>
                            <TableCell>string</TableCell>
                            <TableCell>Fifth sample field</TableCell>
                            <TableCell>"sample_value"</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                  
                  <div className="mb-4">
                    <h4 className="font-medium text-slate-800 mb-2">Sample Data Preview</h4>
                    <div className="bg-white border border-slate-200 rounded-lg overflow-auto">
                      <Table>
                        <TableHeader className="bg-slate-50">
                          <TableRow>
                            <TableHead>field_1</TableHead>
                            <TableHead>field_2</TableHead>
                            <TableHead>field_3</TableHead>
                            <TableHead>field_4</TableHead>
                            <TableHead>field_5</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableRow>
                            <TableCell>value1</TableCell>
                            <TableCell>4.2</TableCell>
                            <TableCell>856.7</TableCell>
                            <TableCell>15.3</TableCell>
                            <TableCell>sample_1</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>value2</TableCell>
                            <TableCell>5.8</TableCell>
                            <TableCell>1245.3</TableCell>
                            <TableCell>22.7</TableCell>
                            <TableCell>sample_2</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>value3</TableCell>
                            <TableCell>7.1</TableCell>
                            <TableCell>978.2</TableCell>
                            <TableCell>18.9</TableCell>
                            <TableCell>sample_3</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Quality Assessment Tab */}
              {activeTab === "quality" && (
                <div>
                  <h3 className="font-semibold text-lg mb-3">Quality Assessment</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <h4 className="font-medium text-slate-800 mb-3">Completeness</h4>
                      <div className="flex items-center mb-2">
                        <Progress
                          value={98.7}
                          className="w-full h-3"
                          color="bg-green-500"
                        />
                        <div className="ml-3 font-semibold">98.7%</div>
                      </div>
                      <p className="text-sm text-slate-500">Only 0.3% of values are missing across all fields.</p>
                    </div>
                    
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <h4 className="font-medium text-slate-800 mb-3">Accuracy</h4>
                      <div className="flex items-center mb-2">
                        <Progress
                          value={95.2}
                          className="w-full h-3"
                          color="bg-green-500"
                        />
                        <div className="ml-3 font-semibold">95.2%</div>
                      </div>
                      <p className="text-sm text-slate-500">Values have been cross-validated with reference sources.</p>
                    </div>
                    
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <h4 className="font-medium text-slate-800 mb-3">Consistency</h4>
                      <div className="flex items-center mb-2">
                        <Progress
                          value={97.8}
                          className="w-full h-3"
                          color="bg-green-500"
                        />
                        <div className="ml-3 font-semibold">97.8%</div>
                      </div>
                      <p className="text-sm text-slate-500">Data follows consistent formatting and units.</p>
                    </div>
                    
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <h4 className="font-medium text-slate-800 mb-3">Timeliness</h4>
                      <div className="flex items-center mb-2">
                        <Progress
                          value={89.5}
                          className="w-full h-3"
                          color="bg-amber-500"
                        />
                        <div className="ml-3 font-semibold">89.5%</div>
                      </div>
                      <p className="text-sm text-slate-500">Dataset was last updated 3 months ago.</p>
                    </div>
                  </div>
                  
                  <div className="bg-white border border-slate-200 rounded-lg p-4">
                    <h4 className="font-medium text-slate-800 mb-3">Improvement Suggestions</h4>
                    <ul className="space-y-2 text-sm text-slate-600">
                      <li className="flex items-start">
                        <Info className="h-4 w-4 text-primary-500 mt-1 mr-2" />
                        <span>Add more specific geolocation data (latitude/longitude) for better spatial analysis.</span>
                      </li>
                      <li className="flex items-start">
                        <Info className="h-4 w-4 text-primary-500 mt-1 mr-2" />
                        <span>Include additional variables for more comprehensive analysis.</span>
                      </li>
                      <li className="flex items-start">
                        <Info className="h-4 w-4 text-primary-500 mt-1 mr-2" />
                        <span>Consider documenting more methodological details for better reproducibility.</span>
                      </li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
