import Link from "next/link";
import { Bot, Database, Home, PlayCircle, Settings2, Sparkles } from "lucide-react";
import { getCurrentUser } from "@/lib/auth";
import { cn } from "@/lib/utils";

const links = [
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/agent", label: "Agent", icon: Bot },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/jobs", label: "Jobs", icon: Sparkles },
  { href: "/playground", label: "Playground", icon: PlayCircle },
  { href: "/settings", label: "Settings", icon: Settings2 }
] as const;

export async function Sidebar() {
  const user = await getCurrentUser();

  return (
    <aside className="rounded-[2rem] border border-black/10 bg-[#10201d] p-6 text-white shadow-panel">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-white/60">Agentic</p>
        <h1 className="mt-3 font-display text-3xl">Fine-tune studio</h1>
        <p className="mt-3 text-sm text-white/70">
          Validate data, launch jobs, and compare models without leaving one dashboard.
        </p>
      </div>

      <nav className="mt-10 space-y-2">
        {links.map((link) => {
          const Icon = link.icon;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium text-white/75 transition hover:bg-white/10 hover:text-white"
              )}
            >
              <Icon className="h-4 w-4" />
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-10 rounded-3xl bg-white/10 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-white/50">Workspace mode</p>
        <p className="mt-2 text-sm font-semibold">{user?.name ?? "Local Workspace"}</p>
        <p className="mt-1 text-xs text-white/60">{user?.email ?? "local@agentic.app"}</p>
      </div>
    </aside>
  );
}
