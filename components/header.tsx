import Link from "next/link";
import { Plus, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header() {
  return (
    <header className="flex items-center justify-between gap-4 rounded-[1.5rem] border border-black/8 bg-white/70 px-5 py-3.5 shadow-panel backdrop-blur">
      <p className="text-sm font-medium text-black/40 tracking-wide">Fine-Tuning Studio</p>
      <div className="flex shrink-0 items-center gap-2">
        <Link href="/datasets/new">
          <Button variant="ghost" size="sm">
            <Upload className="mr-1.5 h-3.5 w-3.5" />
            Dataset
          </Button>
        </Link>
        <Link href="/jobs/new">
          <Button size="sm">
            <Plus className="mr-1.5 h-3.5 w-3.5" />
            New run
          </Button>
        </Link>
      </div>
    </header>
  );
}
