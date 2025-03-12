import { 
  DatabaseIcon, 
  DownloadIcon, 
  GitMergeIcon, 
  CodeIcon, 
  SearchIcon, 
  CheckCircleIcon 
} from "lucide-react";

const Features = () => {
  return (
    <div className="bg-gray-50 py-12" id="features">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:text-center mb-12">
          <h2 className="text-base text-primary-600 font-semibold tracking-wide uppercase">Features</h2>
          <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
            Advanced Metadata Generation for AI Research
          </p>
          <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
            Enhance your AI research with high-quality, structured metadata that follows best practices.
          </p>
        </div>
        
        <div className="mt-10">
          <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <DatabaseIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">Source Public Datasets</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Automatically discover and source datasets from public repositories, saving time on manual searching.
              </dd>
            </div>
            
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <DownloadIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">Automatic Downloads</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Streamline your workflow with one-click downloads of datasets along with their structured metadata.
              </dd>
            </div>
            
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <GitMergeIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">FAIR Compliance</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Ensure datasets are Findable, Accessible, Interoperable, and Reusable with automated FAIR assessment.
              </dd>
            </div>
            
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <CodeIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">Schema.org Mapping</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Convert dataset metadata to Schema.org format for better discoverability and interoperability.
              </dd>
            </div>
            
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <SearchIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">Semantic Search</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Find exactly what you need with NLP-powered semantic search that understands context and meaning.
              </dd>
            </div>
            
            <div className="relative">
              <dt>
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white">
                  <CheckCircleIcon className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-gray-900">Quality Assessment</p>
              </dt>
              <dd className="mt-2 ml-16 text-base text-gray-500">
                Evaluate dataset quality with automated assessment tools that highlight strengths and weaknesses.
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
};

export default Features;
