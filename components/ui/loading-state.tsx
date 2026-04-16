export function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="rounded-3xl border border-black/10 bg-white/70 p-6 text-sm text-black/60">
      {label}
    </div>
  );
}
