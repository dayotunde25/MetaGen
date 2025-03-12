import * as React from "react";
import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  className?: string;
  color?: "primary" | "amber" | "blue" | "green" | "purple" | "red";
  showPercentage?: boolean;
  height?: "sm" | "md" | "lg";
}

export function ProgressBar({
  value,
  className,
  color = "primary",
  showPercentage = false,
  height = "md",
}: ProgressBarProps) {
  const colorClasses = {
    primary: "bg-primary-500",
    amber: "bg-amber-500",
    blue: "bg-blue-500",
    green: "bg-green-500",
    purple: "bg-purple-500",
    red: "bg-red-500",
  };

  const heightClasses = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  return (
    <div className={cn("relative w-full", className)}>
      <div className={cn("w-full bg-gray-200 rounded-full", heightClasses[height])}>
        <div
          className={cn("rounded-full", colorClasses[color], heightClasses[height])}
          style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }}
        ></div>
      </div>
      {showPercentage && (
        <div className="text-xs text-gray-500 mt-1">{Math.round(value)}% complete</div>
      )}
    </div>
  );
}

export default ProgressBar;
