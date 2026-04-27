"use client";

import type { ComponentType } from "react";
import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Bot, Boxes, Database, LayoutDashboard, PlayCircle, Settings2 } from "lucide-react";
import { cn } from "@/lib/utils";

const primaryLinks = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/jobs", label: "Training runs", icon: Activity },
  { href: "/models", label: "My models", icon: Boxes },
  { href: "/settings", label: "Model settings", icon: Settings2 }
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
    <div className="space-y-0.5">
      {label ? <p className="mb-2 px-4 text-[10px] uppercase tracking-[0.28em] text-white/35">{label}</p> : null}
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
              "flex items-center gap-3 rounded-2xl px-4 py-2.5 text-sm font-medium transition-all",
              isActive
                ? "bg-white/18 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
                : "text-white/60 hover:bg-white/8 hover:text-white/90"
            )}
          >
            <Icon className={cn("h-4 w-4 shrink-0 transition-opacity", isActive ? "opacity-100" : "opacity-55")} />
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
    <nav className="mt-8 space-y-5">
      <NavSection links={primaryLinks} pathname={pathname} />
      <NavSection links={secondaryLinks} pathname={pathname} label="Inference" />
    </nav>
  );
}
