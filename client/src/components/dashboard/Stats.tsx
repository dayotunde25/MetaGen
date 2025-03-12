import { useQuery } from "@tanstack/react-query";

interface StatsData {
  totalDatasets: number;
  fairCompliant: number;
  highQualityMetadata: number;
  apiCalls: number;
}

const Stats = () => {
  const { data, isLoading } = useQuery<StatsData>({
    queryKey: ['/api/stats'],
    refetchInterval: 300000, // Refetch every 5 minutes
  });

  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-4">
          <div className="bg-primary-50 overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Total Datasets
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-primary-600">
                  {isLoading ? (
                    <div className="h-9 w-24 bg-gray-200 animate-pulse rounded"></div>
                  ) : (
                    data?.totalDatasets.toLocaleString() || "0"
                  )}
                </dd>
              </dl>
            </div>
          </div>
          <div className="bg-green-50 overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  FAIR Compliant
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-green-500">
                  {isLoading ? (
                    <div className="h-9 w-24 bg-gray-200 animate-pulse rounded"></div>
                  ) : (
                    data?.fairCompliant.toLocaleString() || "0"
                  )}
                </dd>
              </dl>
            </div>
          </div>
          <div className="bg-purple-50 overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Schema.org Mapped
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-purple-500">
                  {isLoading ? (
                    <div className="h-9 w-24 bg-gray-200 animate-pulse rounded"></div>
                  ) : (
                    data?.highQualityMetadata.toLocaleString() || "0"
                  )}
                </dd>
              </dl>
            </div>
          </div>
          <div className="bg-gray-50 overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  API Calls Today
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-gray-700">
                  {isLoading ? (
                    <div className="h-9 w-24 bg-gray-200 animate-pulse rounded"></div>
                  ) : (
                    data?.apiCalls.toLocaleString() || "0"
                  )}
                </dd>
              </dl>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Stats;
