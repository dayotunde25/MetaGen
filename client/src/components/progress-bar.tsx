import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  status: string;
  height?: string;
}

export default function ProgressBar({ value, status, height = "h-2" }: ProgressBarProps) {
  const getProgressColor = () => {
    switch (status) {
      case "processing":
        return "bg-secondary";
      case "completed":
        return "bg-status-success";
      case "error":
        return "bg-destructive";
      case "queued":
      default:
        return "bg-neutral-medium";
    }
  };

  return (
    <div className={cn("flex mb-3 bg-neutral-light rounded-full overflow-hidden", height)}>
      <div 
        className={cn(getProgressColor())} 
        style={{ width: `${value}%` }}
      ></div>
    </div>
  );
}
