import { cn } from "@/lib/utils";

const pulse = "animate-pulse rounded-2xl bg-black/6";

function ListSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className={cn(pulse, "h-8 w-40")} />
        <div className={cn(pulse, "h-7 w-24")} />
      </div>
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="flex items-center gap-4 rounded-[1.5rem] border border-black/8 bg-white/80 px-5 py-4 shadow-sm"
        >
          <div className={cn(pulse, "h-10 w-10 shrink-0")} />
          <div className="flex-1 space-y-2">
            <div className={cn(pulse, "h-4 w-48")} />
            <div className={cn(pulse, "h-3 w-32")} />
          </div>
          <div className={cn(pulse, "h-6 w-16")} />
        </div>
      ))}
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className={cn(pulse, "h-3 w-32")} />
      <div className="rounded-3xl border border-black/10 bg-panel p-6 shadow-panel space-y-4">
        <div className={cn(pulse, "h-10 w-72")} />
        <div className={cn(pulse, "h-4 w-48")} />
        <div className="grid gap-3 sm:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className={cn(pulse, "h-20")} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function LoadingState({
  label = "Loading...",
  variant = "text"
}: {
  label?: string;
  variant?: "text" | "list" | "detail";
}) {
  if (variant === "list") return <ListSkeleton />;
  if (variant === "detail") return <DetailSkeleton />;
  return (
    <div className="rounded-3xl border border-black/10 bg-white/70 p-6 text-sm text-black/60">
      {label}
    </div>
  );
}
