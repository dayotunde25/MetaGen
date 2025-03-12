import { cn } from "@/lib/utils";
import { X } from "lucide-react";

interface FilterChipProps {
  label: string;
  active?: boolean;
  onSelect?: () => void;
  onRemove?: () => void;
}

export default function FilterChip({ 
  label, 
  active = false, 
  onSelect, 
  onRemove 
}: FilterChipProps) {
  const baseClasses = "inline-flex items-center text-sm px-3 py-1 rounded-full mr-2 mt-2 transition-all";
  
  const classes = cn(
    baseClasses,
    active
      ? "bg-primary bg-opacity-10 text-primary"
      : "bg-neutral-lightest text-neutral-dark cursor-pointer hover:bg-neutral-light"
  );
  
  return (
    <span className={classes} onClick={!active ? onSelect : undefined}>
      {label}
      {active && onRemove && (
        <button 
          className="ml-1 focus:outline-none" 
          onClick={(e) => {
            e.stopPropagation(); 
            onRemove();
          }}
          aria-label={`Remove ${label} filter`}
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </span>
  );
}
