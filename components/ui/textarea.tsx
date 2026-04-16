import type { TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        "min-h-40 w-full rounded-3xl border border-black/10 bg-white px-4 py-3 text-sm outline-none transition placeholder:text-black/40 focus:border-brand",
        props.className
      )}
    />
  );
}
