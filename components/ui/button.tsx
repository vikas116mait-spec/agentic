import type { ButtonHTMLAttributes, PropsWithChildren } from "react";
import { cn } from "@/lib/utils";

type ButtonProps = PropsWithChildren<
  ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: "primary" | "secondary" | "ghost" | "danger";
    size?: "sm" | "md";
  }
>;

export function Button({ className, variant = "primary", size = "md", children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-full font-semibold transition",
        size === "md" && "px-4 py-2 text-sm",
        size === "sm" && "px-3 py-1.5 text-xs",
        variant === "primary" && "bg-brand text-brand-foreground hover:opacity-90",
        variant === "secondary" && "bg-ink text-white hover:bg-black",
        variant === "ghost" && "bg-transparent text-ink hover:bg-black/5",
        variant === "danger" && "bg-danger text-white hover:opacity-90",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
