import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm outline-none ring-0 transition placeholder:text-black/40 focus:border-brand",
        props.className
      )}
    />
  );
}
