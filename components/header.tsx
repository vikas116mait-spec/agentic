import Link from "next/link";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header() {
  return (
    <header className="flex flex-col gap-4 rounded-[2rem] border border-black/10 bg-white/75 p-5 shadow-panel backdrop-blur md:flex-row md:items-center md:justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-black/45">Production-minded MVP</p>
        <h2 className="mt-2 font-display text-3xl">Fine-tuning workflow, without the clutter.</h2>
      </div>
      <div className="flex gap-3">
        <Link href="/datasets/new">
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New dataset
          </Button>
        </Link>
        <Link href="/jobs/new">
          <Button variant="ghost">Start job</Button>
        </Link>
      </div>
    </header>
  );
}
