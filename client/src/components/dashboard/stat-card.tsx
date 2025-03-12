import { 
  Database, 
  CheckCircle, 
  Loader, 
  AlertTriangle, 
  ArrowUp, 
  ArrowDown, 
  Clock 
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface StatCardProps {
  title: string;
  value: string;
  icon: "database" | "check" | "loader" | "alert-triangle";
  change?: number;
  timeEstimate?: string;
  color?: "primary" | "green" | "amber" | "red";
  loading?: boolean;
}

export default function StatCard({
  title,
  value,
  icon,
  change,
  timeEstimate,
  color = "primary",
  loading = false
}: StatCardProps) {
  const getIcon = () => {
    switch (icon) {
      case "database":
        return <Database className="h-5 w-5 text-primary-500" />;
      case "check":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "loader":
        return <Loader className="h-5 w-5 text-amber-500" />;
      case "alert-triangle":
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default:
        return null;
    }
  };
  
  const getBgColor = () => {
    switch (color) {
      case "primary":
        return "bg-blue-50";
      case "green":
        return "bg-green-50";
      case "amber":
        return "bg-amber-50";
      case "red":
        return "bg-red-50";
      default:
        return "bg-blue-50";
    }
  };
  
  const getChangeColor = () => {
    if (change === undefined) return "";
    if (icon === "alert-triangle") {
      // For error stats, down is good
      return change < 0 ? "text-green-500" : "text-red-500";
    }
    // For all other stats, up is good
    return change > 0 ? "text-green-500" : "text-red-500";
  };
  
  return (
    <Card className="bg-white shadow rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500 font-medium">{title}</p>
          {loading ? (
            <Skeleton className="h-8 w-20 mt-1" />
          ) : (
            <p className="text-2xl font-semibold mt-1">{value}</p>
          )}
        </div>
        <div className={`p-3 rounded-full ${getBgColor()}`}>
          {getIcon()}
        </div>
      </div>
      
      {loading ? (
        <Skeleton className="h-5 w-32 mt-4" />
      ) : (
        <div className="flex items-center mt-4 text-sm">
          {change !== undefined && (
            <span className={`flex items-center ${getChangeColor()}`}>
              {change > 0 ? (
                <ArrowUp className="h-4 w-4 mr-1" />
              ) : (
                <ArrowDown className="h-4 w-4 mr-1" />
              )}
              {Math.abs(change)}%
            </span>
          )}
          {timeEstimate && (
            <span className="text-amber-500 flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              {timeEstimate}
            </span>
          )}
          {change !== undefined && (
            <span className="text-gray-500 ml-2">from last month</span>
          )}
        </div>
      )}
    </Card>
  );
}
