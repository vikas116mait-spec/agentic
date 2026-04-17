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
  Upload
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { StatusBadge } from "@/components/status-badge";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate } from "@/lib/utils";

type DashboardSummary = {
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
    title: "Train and export",
    hint: "Run, watch, download.",
    done: (d) => d.totalJobs > 0
  }
];

const pillars = [
  {
    icon: Cpu,
    eyebrow: "Run Locally",
    title: "Use your own GPU first.",
    copy: "Train on your machine."
  },
  {
    icon: FileStack,
    eyebrow: "No-Code Training",
    title: "Dataset, model, start.",
    copy: "Keep the flow simple."
  },
  {
    icon: Download,
    eyebrow: "Export Models",
    title: "Take the trained result with you.",
    copy: "Download the adapter."
  }
];

export function DashboardClient() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<DashboardSummary>("/dashboard/summary")
      .then(setData)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) {
    return <ErrorAlert title="Cannot reach the API" description={`Start the Python service on port 8001. ${error}`} />;
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
              Train a custom model locally without getting lost in setup.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-black/62 sm:text-base">Model. Data. Train. Export.</p>

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
              <p className="text-xs uppercase tracking-[0.2em] text-white/55">Recommended model families</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {[
                  "Qwen 0.5B / 1.5B / 3B / 7B",
                  "Gemma 2 2B",
                  "Phi 3.5 Mini",
                  "Mistral 7B",
                  "Llama 3.2 3B"
                ].map((label) => (
                  <span key={label} className="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs text-white/80">
                    {label}
                  </span>
                ))}
              </div>
              <p className="mt-4 text-sm leading-6 text-white/68">Start small. Scale up later.</p>
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
        <Card className="space-y-3 bg-white/88">
          <p className="font-display text-2xl">Recommended next step</p>
          <p className="text-sm leading-7 text-black/56">
            {data.totalDatasets === 0
              ? "Upload a dataset."
              : data.totalJobs === 0
                ? "Start the first run."
                : "Open the latest run."}
          </p>
          <div>
            <Link href={data.totalDatasets === 0 ? "/datasets/new" : "/jobs/new"}>
              <Button>{data.totalDatasets === 0 ? "Upload dataset" : "Start training"}</Button>
            </Link>
          </div>
        </Card>
      ) : null}

      <Card className="bg-white/88">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-display text-2xl">Recent training runs</p>
            <p className="mt-2 text-sm text-black/55">Latest runs.</p>
          </div>
          {data.latestActivity.length > 0 ? (
            <Link href="/jobs">
              <Button variant="ghost" size="sm">
                View all
              </Button>
            </Link>
          ) : null}
        </div>

        {data.latestActivity.length === 0 ? (
          <p className="mt-4 text-sm text-black/50">No runs yet. Upload a dataset, then start fine-tuning.</p>
        ) : (
          <div className="mt-5 space-y-2">
            {data.latestActivity.map((job) => (
              <Link key={job.id} href={`/jobs/${job.id}`} className="block">
                <div className="flex items-center justify-between rounded-[1.4rem] border border-black/6 bg-canvas/75 px-4 py-3 transition hover:bg-canvas">
                  <div className="min-w-0">
                    <p className="truncate font-medium">{job.datasetName}</p>
                    <p className="truncate text-xs text-black/45">
                      {job.baseModel} · {formatDate(job.createdAt)}
                    </p>
                  </div>
                  <StatusBadge value={job.status} />
                </div>
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
