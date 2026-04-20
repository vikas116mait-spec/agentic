import { getCurrentUser } from "@/lib/auth";
import { NavLinks } from "@/components/nav-links";

export async function Sidebar() {
  const user = await getCurrentUser();

  return (
    <aside className="flex flex-col rounded-[2rem] border border-black/10 bg-[#10201d] p-6 text-white shadow-panel">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-white/50">Simple fine-tuning</p>
        <h1 className="mt-3 font-display text-3xl">Agentic</h1>
        <p className="mt-3 text-sm text-white/65">
          Pick a model, upload data, and start training. Everything else is optional.
        </p>
      </div>

      <NavLinks />

      <div className="mt-auto pt-10">
        <div className="rounded-3xl bg-white/10 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-white/50">Workspace</p>
          <p className="mt-2 text-sm font-semibold">{user?.name ?? "Local workspace"}</p>
          <p className="mt-1 text-xs text-white/55">{user?.email ?? "local@agentic.app"}</p>
        </div>
      </div>
    </aside>
  );
}
