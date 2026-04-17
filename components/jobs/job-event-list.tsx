import { Card } from "@/components/ui/card";
import { formatDate } from "@/lib/utils";

export function JobEventList({
  events
}: {
  events: Array<{ id: string; level: string; message: string; createdAt: Date; eventType: string | null }>;
}) {
  return (
    <Card className="space-y-4">
      <div>
        <p className="font-display text-2xl">Events</p>
        <p className="mt-2 text-sm text-black/60">Live training events and sync history for this run.</p>
      </div>

      <div className="space-y-3">
        {events.length === 0 ? <p className="text-sm text-black/55">No events synced yet.</p> : null}
        {events.map((event) => (
          <div key={event.id} className="rounded-2xl bg-white p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-muted px-2 py-1 text-xs font-semibold uppercase">{event.level}</span>
              {event.eventType ? <span className="text-xs text-black/45">{event.eventType}</span> : null}
              <span className="text-xs text-black/45">{formatDate(event.createdAt)}</span>
            </div>
            <p className="mt-3 text-sm text-black/75">{event.message}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}
