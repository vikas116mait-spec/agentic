import { Cpu, Download, FileStack, Sparkles } from "lucide-react";
import { CreateJobForm } from "@/components/jobs/create-job-form";
import { Card } from "@/components/ui/card";

const highlights = [
  {
    icon: Cpu,
    eyebrow: "Local-first",
    title: "Train on your own GPU",
    copy: "Use your GPU."
  },
  {
    icon: FileStack,
    eyebrow: "Simple flow",
    title: "Dataset, model, start",
    copy: "Keep it simple."
  },
  {
    icon: Download,
    eyebrow: "Export",
    title: "Download the result",
    copy: "Download the adapter."
  }
];

export default async function NewJobPage({
  searchParams
}: {
  searchParams: Promise<{ datasetId?: string }>;
}) {
  const params = await searchParams;

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden p-0">
        <div className="grid gap-0 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(255,247,229,0.9))] p-7 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-brand/15 bg-brand/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand">
              <Sparkles className="h-3.5 w-3.5" />
              Start Fine-Tuning
            </div>

            <h1 className="mt-5 max-w-3xl font-display text-4xl leading-tight sm:text-5xl">
              Train a custom model with one focused workflow.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-black/62 sm:text-base">Pick model. Pick dataset. Start.</p>

            <div className="mt-6 flex flex-wrap gap-2">
              {[
                "Qwen 0.5B / 1.5B / 3B / 7B",
                "Gemma 2 2B",
                "Phi 3.5 Mini",
                "Mistral 7B",
                "Llama 3.2 3B"
              ].map((label) => (
                <span key={label} className="rounded-full border border-black/8 bg-white/80 px-3 py-1 text-xs text-black/70">
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-[linear-gradient(180deg,rgba(12,29,28,0.98),rgba(25,64,57,0.94))] p-7 text-white sm:p-8">
            <p className="text-xs uppercase tracking-[0.22em] text-white/55">How this page works</p>
            <div className="mt-4 space-y-3">
              {[
                "1. Pick dataset",
                "2. Pick model",
                "3. Start run",
                "4. Download result"
              ].map((step) => (
                <div key={step} className="rounded-[1.3rem] border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                  {step}
                </div>
              ))}
            </div>
            <p className="mt-5 text-sm leading-7 text-white/68">Start with a small model.</p>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        {highlights.map((highlight) => {
          const Icon = highlight.icon;

          return (
            <Card key={highlight.title} className="space-y-4 bg-white/88">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-black/42">{highlight.eyebrow}</p>
                <p className="mt-2 font-display text-2xl">{highlight.title}</p>
              </div>
              <p className="text-sm leading-7 text-black/58">{highlight.copy}</p>
            </Card>
          );
        })}
      </div>

      <Card className="space-y-6 bg-white/90">
        <div>
          <p className="font-display text-3xl">Launch a new run</p>
          <p className="mt-2 text-sm text-black/60">Use local QLoRA for the simplest path.</p>
        </div>

        <CreateJobForm initialDatasetId={params.datasetId ?? ""} />
      </Card>
    </div>
  );
}
