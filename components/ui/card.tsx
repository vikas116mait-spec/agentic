import type { HTMLAttributes, PropsWithChildren } from "react";
import { cn } from "@/lib/utils";

type CardProps = PropsWithChildren<HTMLAttributes<HTMLDivElement>>;

export function Card({ className, children, ...props }: CardProps) {
  return (
    <div className={cn("rounded-3xl border border-black/10 bg-panel p-6 shadow-panel", className)} {...props}>
      {children}
    </div>
  );
}
