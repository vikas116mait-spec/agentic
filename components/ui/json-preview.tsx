export function JsonPreview({ value }: { value: unknown }) {
  return (
    <pre className="overflow-x-auto rounded-3xl bg-ink p-4 text-xs text-white">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
