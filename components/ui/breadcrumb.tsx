"use client";

import type { ComponentProps } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

type Crumb = { label: string; href?: ComponentProps<typeof Link>["href"] };

export function Breadcrumb({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav className="flex items-center gap-1.5 text-xs text-black/45">
      {crumbs.map((crumb, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && <ChevronRight className="h-3 w-3 shrink-0 opacity-40" />}
          {crumb.href ? (
            <Link href={crumb.href} className={cn("hover:text-black/70 transition")}>
              {crumb.label}
            </Link>
          ) : (
            <span className="font-medium text-black/70">{crumb.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
