import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: number;
  icon: string;
  change?: { value: number; isPositive: boolean };
  color: "primary" | "secondary" | "accent" | "info";
}

export default function StatsCard({ title, value, icon, change, color }: StatsCardProps) {
  const colorMap = {
    primary: "border-primary",
    secondary: "border-secondary",
    accent: "border-accent",
    info: "border-status-info",
  };

  const iconColorMap = {
    primary: "text-primary-light",
    secondary: "text-secondary-light",
    accent: "text-accent-light",
    info: "text-status-info",
  };

  return (
    <Card className={cn("border-l-4", colorMap[color])}>
      <CardContent className="p-4">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-neutral-medium text-sm font-medium mb-1">{title}</h3>
            <p className="text-2xl font-bold">{value}</p>
          </div>
          <span className={cn("material-icons text-3xl", iconColorMap[color])}>{icon}</span>
        </div>
        
        {change && (
          <p className="text-xs text-neutral-medium mt-2">
            <span className={change.isPositive ? "text-status-success" : "text-status-error"}>
              <span className="material-icons text-sm">
                {change.isPositive ? "arrow_upward" : "arrow_downward"}
              </span>
              {" "}
              {change.value}%
            </span>
            {" since last month"}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
