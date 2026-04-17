import Link from "next/link";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header() {
  return (
    <header className="flex flex-col gap-4 rounded-[2rem] border border-black/10 bg-white/75 p-5 shadow-panel backdrop-blur md:flex-row md:items-center md:justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-black/40">Model. Data. Train.</p>
        <h2 className="mt-1.5 font-display text-2xl">Start a fine-tune without digging through extra screens.</h2>
      </div>
      <div className="flex shrink-0 gap-3">
        <Link href="/jobs/new">
          <Button size="sm">
            <Plus className="mr-1.5 h-3.5 w-3.5" />
            Start fine-tuning
          </Button>
        </Link>
        <Link href="/datasets/new">
          <Button variant="ghost" size="sm">Upload dataset</Button>
        </Link>
      </div>
    </header>
  );
}
