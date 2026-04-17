"use client";

import type { ComponentType } from "react";
import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bot, Database, PlayCircle, Settings2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const primaryLinks = [
  { href: "/dashboard", label: "Start here", icon: Sparkles },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/jobs", label: "Training runs", icon: Sparkles },
  { href: "/settings", label: "Models", icon: Settings2 }
] as const;

const secondaryLinks = [
  { href: "/playground", label: "Playground", icon: PlayCircle },
  { href: "/agent", label: "Agent", icon: Bot }
] as const;

function NavSection({
  links,
  pathname,
  label
}: {
  links: ReadonlyArray<{ href: Route; label: string; icon: ComponentType<{ className?: string }> }>;
  pathname: string;
  label?: string;
}) {
  return (
    <div className="space-y-1">
      {label ? <p className="px-4 text-[11px] uppercase tracking-[0.25em] text-white/40">{label}</p> : null}
      {links.map((link) => {
        const Icon = link.icon;
        const isActive =
          pathname === link.href ||
          (link.href !== "/dashboard" && pathname.startsWith(link.href));

        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              "flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium transition-colors",
              isActive
                ? "bg-white/20 text-white"
                : "text-white/70 hover:bg-white/10 hover:text-white"
            )}
          >
            <Icon className={cn("h-4 w-4 shrink-0", isActive ? "opacity-100" : "opacity-70")} />
            {link.label}
          </Link>
        );
      })}
    </div>
  );
}

export function NavLinks() {
  const pathname = usePathname();

  return (
    <nav className="mt-10 space-y-6">
      <NavSection links={primaryLinks} pathname={pathname} />
      <NavSection links={secondaryLinks} pathname={pathname} label="Secondary" />
    </nav>
  );
}
