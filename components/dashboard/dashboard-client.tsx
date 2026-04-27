"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Download,
  FileStack,
  Settings2,
  Sparkles,
  Upload,
  Zap
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { StatusBadge } from "@/components/status-badge";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate } from "@/lib/utils";

export type DashboardSummary = {
  totalDatasets: number;
  totalJobs: number;
  runningJobs: number;
  completedModels: number;
  latestActivity: Array<{
    id: string;
    datasetName: string;
    baseModel: string;
    status: string;
    createdAt: string;
  }>;
};

type Step = {
  icon: typeof Upload;
  title: string;
  hint: string;
  done: (d: DashboardSummary) => boolean;
};

const steps: Step[] = [
  {
    icon: Settings2,
    title: "Pick a base model",
    hint: "Start small.",
    done: () => true
  },
  {
    icon: Upload,
    title: "Upload one dataset",
    hint: "One JSONL file.",
    done: (d) => d.totalDatasets > 0
  },
  {
    icon: CheckCircle2,
    title: "Train and export to GGUF",
    hint: "Run with Unsloth, watch live loss, export.",
    done: (d) => d.totalJobs > 0
  }
];

const pillars = [
  {
    icon: Zap,
    eyebrow: "Unsloth Accelerated",
    title: "2x faster. 70% less VRAM.",
    copy: "Install Unsloth and your local training runs are automatically accelerated — no config changes needed."
  },
  {
    icon: FileStack,
    eyebrow: "No-Code Training",
    title: "Dataset, model, start.",
    copy: "Upload a JSONL dataset, pick a base model, and hit train. SFT, LoRA, and QLoRA all work out of the box."
  },
  {
    icon: Download,
    eyebrow: "Export Anywhere",
    title: "GGUF → Ollama in one step.",
    copy: "After training, export to GGUF and push directly to your local Ollama — instantly usable in the Playground."
  }
];

export function DashboardClient({
  initialData = null,
  initialError = null,
}: {
  initialData?: DashboardSummary | null;
  initialError?: string | null;
}) {
  const [data, setData] = useState<DashboardSummary | null>(initialData);
  const [error, setError] = useState<string | null>(initialError);

  useEffect(() => {
    pythonApiFetch<DashboardSummary>("/dashboard/summary")
      .then((payload) => {
        setData(payload);
        setError(null);
      })
      .catch((requestError: Error) => {
        if (!initialData) {
          setError(requestError.message);
        }
      });
  }, [initialData]);

  if (error) {
    return (
      <ErrorAlert
        title="Cannot reach the API"
        description={
          `${error} Make sure the Python FastAPI service is running and that ` +
          `PYTHON_API_URL in your Next.js server environment points at it.`
        }
      />
    );
  }

  if (!data) {
    return <LoadingState label="Loading fine-tuning studio..." />;
  }

  const allDone = steps.every((step) => step.done(data));

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden p-0">
        <div className="grid gap-0 lg:grid-cols-[1.35fr_0.95fr]">
          <div className="bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(255,247,229,0.88))] p-7 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-brand/15 bg-brand/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand">
              <Sparkles className="h-3.5 w-3.5" />
              Fine-Tuning Studio
            </div>

            <h1 className="mt-5 max-w-3xl font-display text-4xl leading-tight sm:text-5xl">
              Fine-tune locally with Unsloth — 2x faster, 70% less VRAM.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-black/62 sm:text-base">
              Upload data. Pick a model. Train with Unsloth. Export to GGUF. Run in Ollama.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/jobs/new">
                <Button>
                  Start fine-tuning
                  <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
                </Button>
              </Link>
              <Link href="/datasets/new">
                <Button variant="ghost">Upload dataset</Button>
              </Link>
              <Link href="/settings">
                <Button variant="ghost">Browse models</Button>
              </Link>
            </div>

            <div className="mt-8 grid gap-3 md:grid-cols-3">
              {steps.map((step) => {
                const Icon = step.icon;
                const done = step.done(data);

                return (
                  <div
                    key={step.title}
                    className={`rounded-[1.6rem] border p-5 ${done ? "border-brand/15 bg-brand/5" : "border-black/8 bg-white/90"}`}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`flex h-11 w-11 items-center justify-center rounded-2xl ${
                          done ? "bg-brand text-white" : "bg-black/5 text-black/55"
                        }`}
                      >
                        <Icon className="h-4.5 w-4.5" />
                      </div>
                      <div>
                        <p className="font-semibold">{step.title}</p>
                        <p className="text-xs uppercase tracking-[0.18em] text-black/38">{done ? "Ready" : "Next"}</p>
                      </div>
                    </div>
                    <p className="mt-4 text-sm leading-6 text-black/58">{step.hint}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-[linear-gradient(180deg,rgba(12,29,28,0.98),rgba(25,64,57,0.94))] p-7 text-white sm:p-8">
            <p className="text-xs uppercase tracking-[0.25em] text-white/55">Studio at a glance</p>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {[
                { label: "Datasets", value: data.totalDatasets },
                { label: "Training runs", value: data.totalJobs },
                { label: "Running now", value: data.runningJobs },
                { label: "Completed models", value: data.completedModels }
              ].map((stat) => (
                <div key={stat.label} className="rounded-[1.4rem] border border-white/10 bg-white/5 p-4 backdrop-blur">
                  <p className="text-sm text-white/60">{stat.label}</p>
                  <p className="mt-2 font-display text-3xl">{stat.value}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-[1.8rem] border border-white/10 bg-white/5 p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-white/55">Unsloth-supported models</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {[
                  "Qwen 2.5 0.5B–72B",
                  "Mistral v0.3",
                  "Phi 3.5 Mini",
                  "DeepSeek-R1",
                  "Qwen 3"
                ].map((label) => (
                  <span key={label} className="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs text-white/80">
                    {label}
                  </span>
                ))}
              </div>
              <p className="mt-4 text-sm leading-6 text-white/68">
                500+ models supported. Start with 3B, scale up when ready.
              </p>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        {pillars.map((pillar) => {
          const Icon = pillar.icon;

          return (
            <Card key={pillar.title} className="space-y-4 bg-white/88">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-black/42">{pillar.eyebrow}</p>
                <p className="mt-2 font-display text-2xl">{pillar.title}</p>
              </div>
              <p className="text-sm leading-7 text-black/58">{pillar.copy}</p>
            </Card>
          );
        })}
      </div>

      {!allDone ? (
        <Card className="flex items-center justify-between gap-4 bg-white/88">
          <div>
            <p className="font-display text-xl">
              {data.totalDatasets === 0 ? "Upload your first dataset" : "Start your first training run"}
            </p>
            <p className="mt-1 text-sm text-black/50">
              {data.totalDatasets === 0
                ? "Bring a .jsonl file or enter data manually."
                : "Pick a dataset and a base model to begin."}
            </p>
          </div>
          <Link href={data.totalDatasets === 0 ? "/datasets/new" : "/jobs/new"} className="shrink-0">
            <Button>{data.totalDatasets === 0 ? "Upload dataset" : "Start training"}</Button>
          </Link>
        </Card>
      ) : null}

      <Card className="bg-white/88">
        <div className="flex items-center justify-between">
          <p className="font-display text-2xl">Recent training runs</p>
          {data.latestActivity.length > 0 ? (
            <Link href="/jobs">
              <Button variant="ghost" size="sm">View all</Button>
            </Link>
          ) : null}
        </div>

        {data.latestActivity.length === 0 ? (
          <p className="mt-4 text-sm text-black/50">No runs yet. Upload a dataset, then start fine-tuning.</p>
        ) : (
          <div className="mt-4 divide-y divide-black/5 overflow-hidden rounded-2xl border border-black/8">
            {data.latestActivity.map((job) => (
              <Link key={job.id} href={`/jobs/${job.id}`} className="group flex items-center justify-between gap-4 bg-white/60 px-4 py-3 transition hover:bg-white">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{job.datasetName}</p>
                  <p className="truncate text-xs text-black/40 mt-0.5">
                    {job.baseModel} · {formatDate(job.createdAt)}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <StatusBadge value={job.status} />
                  <ArrowRight className="h-3.5 w-3.5 text-black/20 transition group-hover:translate-x-0.5 group-hover:text-black/40" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
