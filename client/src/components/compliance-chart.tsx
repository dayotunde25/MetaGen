import { Card, CardContent } from "@/components/ui/card";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from "recharts";

interface ComplianceMetric {
  name: string;
  value: number;
  category?: string;
}

interface ComplianceChartProps {
  title: string;
  metrics: ComplianceMetric[];
  overallScore: number;
  color?: string;
}

export default function ComplianceChart({ 
  title, 
  metrics, 
  overallScore,
  color = "#3949ab" 
}: ComplianceChartProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "#4caf50"; // success
    if (score >= 60) return "#2196f3"; // info
    if (score >= 40) return "#ff9800"; // warning
    return "#f44336"; // error
  };
  
  return (
    <Card className="border border-neutral-light h-full">
      <CardContent className="p-4">
        <h3 className="font-medium text-lg mb-4">{title}</h3>
        
        <div className="space-y-4">
          {metrics.map((metric, index) => (
            <div key={index} className="flex items-center justify-between">
              <span className="font-medium">{metric.name}</span>
              <div className="w-48 bg-neutral-light rounded-full h-2">
                <div 
                  className={`h-2 rounded-full`}
                  style={{ 
                    width: `${metric.value}%`, 
                    backgroundColor: getScoreColor(metric.value)
                  }}
                ></div>
              </div>
              <span 
                className="font-medium"
                style={{ color: getScoreColor(metric.value) }}
              >
                {metric.value}%
              </span>
            </div>
          ))}
          
          <div className="flex items-center justify-between mt-6 pt-4 border-t border-neutral-light">
            <span className="font-medium">Overall {title.split(' ')[0]} Score</span>
            <div className="w-48 bg-neutral-light rounded-full h-2">
              <div 
                className={`h-2 rounded-full`}
                style={{ 
                  width: `${overallScore}%`, 
                  backgroundColor: getScoreColor(overallScore)
                }}
              ></div>
            </div>
            <span 
              className="font-medium"
              style={{ color: getScoreColor(overallScore) }}
            >
              {overallScore}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
