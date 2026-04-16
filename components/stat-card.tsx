import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";

export function StatCard({
  label,
  value,
  detail,
  icon
}: {
  label: string;
  value: string | number;
  detail?: string;
  icon?: ReactNode;
}) {
  return (
    <Card className="bg-white">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-black/55">{label}</p>
          <p className="mt-3 font-display text-4xl">{value}</p>
          {detail ? <p className="mt-3 text-sm text-black/60">{detail}</p> : null}
        </div>
        {icon ? <div className="rounded-2xl bg-muted p-3">{icon}</div> : null}
      </div>
    </Card>
  );
}
