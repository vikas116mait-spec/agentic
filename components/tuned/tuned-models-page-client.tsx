"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  ChevronUp,
  Download,
  ExternalLink,
  MessageCircle,
  PlayCircle,
  Save,
  Sparkles,
  Zap
} from "lucide-react";
import { TestModelPanel } from "@/components/jobs/test-model-panel";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { getPythonApiBaseUrl, pythonApiFetch } from "@/lib/python-api";
import type { ModelProfile, ModelProfilesResponse } from "@/lib/types";
import { formatDate } from "@/lib/utils";

type TunedJob = {
  id: string;
  status: string;
  baseModel: string;
  fineTunedModel: string | null;
  modelProvider: string;
  modelProviderLabel: string;
  datasetName: string;
  trainingBackend: string | null;
  createdAt: string;
  ollamaModelName: string | null;
  progressJson?: {
    unslothActive?: boolean | null;
    ollamaRegistered?: boolean | null;
    metrics?: {
      summary?: {
        evalLoss?: number | null;
        perplexity?: number | null;
      } | null;
    } | null;
  } | null;
  resultFilesJson?: Array<{ type: string; path: string }> | null;
};

type SaveStatus = "idle" | "saving" | "saved" | "error";

type ProfileTarget = {
  provider: string;
  model: string;
  suggestedName: string;
  canSave: boolean;
  reason: string | null;
};

function resolveProfileTarget(job: TunedJob): ProfileTarget {
  const suggestedName = `${job.datasetName} · ${job.id.slice(0, 8)}`;

  if (job.modelProvider === "local") {
    const ollamaName = (job.ollamaModelName ?? "").trim();
    if (!ollamaName) {
      return {
        provider: "ollama",
        model: "",
        suggestedName,
        canSave: false,
        reason:
          "Register this adapter with Ollama (see the job page) before saving it as a profile. Local adapters must run through Ollama to be callable from the playground."
      };
    }
    return {
      provider: "ollama",
      model: ollamaName,
      suggestedName,
      canSave: true,
      reason: null
    };
  }

  if (!job.fineTunedModel) {
    return {
      provider: job.modelProvider,
      model: "",
      suggestedName,
      canSave: false,
      reason: "This run did not report a fine-tuned model id yet."
    };
  }

  return {
    provider: job.modelProvider,
    model: job.fineTunedModel,
    suggestedName,
    canSave: true,
    reason: null
  };
}

function findSavedProfile(profiles: ModelProfile[], target: ProfileTarget): ModelProfile | null {
  if (!target.canSave) {
    return null;
  }
  return (
    profiles.find(
      (profile) => profile.provider === target.provider && profile.model === target.model
    ) ?? null
  );
}

function formatDecimal(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }
  return value.toFixed(digits);
}

export function TunedModelsPageClient() {
  const router = useRouter();
  const [jobs, setJobs] = useState<TunedJob[] | null>(null);
  const [profilesData, setProfilesData] = useState<ModelProfilesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openChatJobId, setOpenChatJobId] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<Record<string, SaveStatus>>({});
  const [saveMessage, setSaveMessage] = useState<Record<string, string | null>>({});

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      pythonApiFetch<TunedJob[]>("/jobs"),
      pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles")
    ])
      .then(([jobsPayload, profilesPayload]) => {
        if (cancelled) {
          return;
        }
        setJobs(jobsPayload);
        setProfilesData(profilesPayload);
      })
      .catch((requestError: Error) => {
        if (!cancelled) {
          setError(requestError.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const tunedJobs = useMemo(() => {
    if (!jobs) {
      return null;
    }
    return jobs.filter(
      (job) => ["succeeded", "completed"].includes(job.status) && Boolean(job.fineTunedModel)
    );
  }, [jobs]);

  async function saveAsProfile(job: TunedJob): Promise<ModelProfile | null> {
    const target = resolveProfileTarget(job);
    if (!target.canSave) {
      setSaveStatus((prev) => ({ ...prev, [job.id]: "error" }));
      setSaveMessage((prev) => ({ ...prev, [job.id]: target.reason }));
      return null;
    }

    const existing = findSavedProfile(profilesData?.profiles ?? [], target);
    if (existing) {
      setSaveStatus((prev) => ({ ...prev, [job.id]: "saved" }));
      setSaveMessage((prev) => ({ ...prev, [job.id]: "Already saved in Model settings." }));
      return existing;
    }

    setSaveStatus((prev) => ({ ...prev, [job.id]: "saving" }));
    setSaveMessage((prev) => ({ ...prev, [job.id]: null }));

    try {
      const created = await pythonApiFetch<ModelProfile>("/settings/model-profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: target.suggestedName,
          provider: target.provider,
          model: target.model,
          category: "custom",
          description: `Fine-tuned on ${job.datasetName} (base ${job.baseModel}).`
        })
      });

      setProfilesData((prev) =>
        prev
          ? { ...prev, profiles: [...prev.profiles, created] }
          : { profiles: [created], defaults: {
              playgroundBaseProfileId: null,
              playgroundCompareProfileId: null,
              agentBaseProfileId: null,
              agentModelProfileId: null,
              jobBaseProfileId: null
            }, providers: [] }
      );
      setSaveStatus((prev) => ({ ...prev, [job.id]: "saved" }));
      setSaveMessage((prev) => ({ ...prev, [job.id]: "Saved to Model settings." }));
      return created;
    } catch (requestError) {
      setSaveStatus((prev) => ({ ...prev, [job.id]: "error" }));
      setSaveMessage((prev) => ({
        ...prev,
        [job.id]: requestError instanceof Error ? requestError.message : "Save failed."
      }));
      return null;
    }
  }

  async function openInPlayground(job: TunedJob) {
    const target = resolveProfileTarget(job);
    if (!target.canSave) {
      setSaveStatus((prev) => ({ ...prev, [job.id]: "error" }));
      setSaveMessage((prev) => ({ ...prev, [job.id]: target.reason }));
      return;
    }
    const existing = findSavedProfile(profilesData?.profiles ?? [], target);
    const profile = existing ?? (await saveAsProfile(job));
    if (!profile) {
      return;
    }
    router.push(`/playground?compareProfileId=${encodeURIComponent(profile.id)}`);
  }

  if (error) {
    return <ErrorAlert title="Could not load tuned models" description={error} />;
  }

  if (!jobs || !profilesData) {
    return <LoadingState variant="list" />;
  }

  if (!tunedJobs || tunedJobs.length === 0) {
    return (
      <EmptyState
        title="No tuned models yet"
        description="Completed fine-tuning runs will appear here so you can chat with them, save them as reusable profiles, and jump into the playground."
        action={
          <div className="flex gap-3">
            <Link href="/jobs/new">
              <Button>Start a fine-tuning run</Button>
            </Link>
            <Link href="/jobs">
              <Button variant="ghost">View training runs</Button>
            </Link>
          </div>
        }
      />
    );
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Your fine-tuned models</p>
            <p className="font-display text-3xl">Tuned models</p>
            <p className="mt-2 text-sm text-black/60">
              Every completed fine-tuning run lives here. Chat inline, save it as a reusable profile, or open it in the playground against any base model.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded-full bg-brand/10 px-3 py-1 text-xs font-semibold text-brand">
              {tunedJobs.length} ready
            </span>
            <Link href="/jobs/new">
              <Button size="sm">Start new run</Button>
            </Link>
          </div>
        </div>
      </Card>

      <div className="space-y-4">
        {tunedJobs.map((job) => {
          const target = resolveProfileTarget(job);
          const savedProfile = findSavedProfile(profilesData.profiles, target);
          const status = saveStatus[job.id] ?? (savedProfile ? "saved" : "idle");
          const message = saveMessage[job.id] ?? null;
          const ggufFile = job.resultFilesJson?.find((file) => file.type === "local_gguf") ?? null;
          const evalLoss = formatDecimal(job.progressJson?.metrics?.summary?.evalLoss ?? null, 4);
          const perplexity = formatDecimal(job.progressJson?.metrics?.summary?.perplexity ?? null, 2);
          const unslothActive = job.progressJson?.unslothActive ?? null;
          const ollamaRegistered = job.progressJson?.ollamaRegistered ?? false;
          const chatOpen = openChatJobId === job.id;
          const canDownloadLocal = job.modelProvider === "local";
          const downloadUrl = `${getPythonApiBaseUrl()}/jobs/${job.id}/download`;
          const ggufDownloadUrl = `${getPythonApiBaseUrl()}/jobs/${job.id}/download?type=gguf`;

          return (
            <Card key={job.id} className="space-y-4 bg-white/88">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="truncate font-display text-xl text-black/90">{job.datasetName}</p>
                    <StatusBadge value={job.status} />
                    {unslothActive === true ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-brand/10 px-2 py-0.5 text-xs font-semibold text-brand">
                        <Zap className="h-3 w-3" />
                        Unsloth
                      </span>
                    ) : null}
                    {ggufFile ? (
                      <span className="inline-flex items-center rounded-full bg-accent/20 px-2 py-0.5 text-xs font-semibold text-black/65">
                        GGUF
                      </span>
                    ) : null}
                    {job.modelProvider === "local" && ollamaRegistered ? (
                      <span className="inline-flex items-center rounded-full bg-brand/10 px-2 py-0.5 text-xs font-semibold text-brand">
                        Ollama ready
                      </span>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                    <span className="text-xs text-black/45">{job.modelProviderLabel}</span>
                    <span className="text-xs text-black/20">·</span>
                    <span className="truncate font-mono text-xs text-black/55">{job.baseModel}</span>
                    <span className="text-xs text-black/20">·</span>
                    <span className="text-xs text-black/45">Completed {formatDate(job.createdAt)}</span>
                  </div>
                  {job.fineTunedModel ? (
                    <p className="truncate font-mono text-xs text-brand/80">{job.fineTunedModel}</p>
                  ) : null}
                  {job.modelProvider === "local" && job.ollamaModelName ? (
                    <p className="truncate text-xs text-black/55">
                      Ollama name: <span className="font-mono text-black/70">{job.ollamaModelName}</span>
                    </p>
                  ) : null}
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3 lg:grid-cols-2 lg:text-right">
                  {evalLoss ? (
                    <div className="rounded-xl bg-muted/60 px-3 py-2 text-left">
                      <p className="text-[10px] uppercase tracking-wide text-black/40">Eval loss</p>
                      <p className="mt-0.5 font-semibold text-black/80">{evalLoss}</p>
                    </div>
                  ) : null}
                  {perplexity ? (
                    <div className="rounded-xl bg-muted/60 px-3 py-2 text-left">
                      <p className="text-[10px] uppercase tracking-wide text-black/40">Perplexity</p>
                      <p className="mt-0.5 font-semibold text-black/80">{perplexity}</p>
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  variant={chatOpen ? "secondary" : "primary"}
                  onClick={() => setOpenChatJobId(chatOpen ? null : job.id)}
                >
                  {chatOpen ? (
                    <>
                      <ChevronUp className="mr-1.5 h-4 w-4" />
                      Hide chat
                    </>
                  ) : (
                    <>
                      <MessageCircle className="mr-1.5 h-4 w-4" />
                      Chat
                    </>
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={!target.canSave || status === "saving" || status === "saved"}
                  onClick={() => void saveAsProfile(job)}
                >
                  {status === "saved" ? (
                    <>
                      <CheckCircle2 className="mr-1.5 h-4 w-4" />
                      Saved
                    </>
                  ) : status === "saving" ? (
                    "Saving..."
                  ) : (
                    <>
                      <Save className="mr-1.5 h-4 w-4" />
                      Save as profile
                    </>
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={!target.canSave || status === "saving"}
                  onClick={() => void openInPlayground(job)}
                >
                  <PlayCircle className="mr-1.5 h-4 w-4" />
                  Open in playground
                </Button>
                {canDownloadLocal ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => window.location.assign(downloadUrl)}
                  >
                    <Download className="mr-1.5 h-4 w-4" />
                    Adapter
                  </Button>
                ) : null}
                {ggufFile ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => window.location.assign(ggufDownloadUrl)}
                  >
                    <Download className="mr-1.5 h-4 w-4" />
                    GGUF
                  </Button>
                ) : null}
                <Link href={`/jobs/${job.id}`}>
                  <Button size="sm" variant="ghost">
                    <ExternalLink className="mr-1.5 h-4 w-4" />
                    Open run
                  </Button>
                </Link>
              </div>

              {message ? (
                <p
                  className={`text-xs ${
                    status === "error" ? "text-danger" : "text-black/55"
                  }`}
                >
                  {message}
                </p>
              ) : null}

              {!target.canSave && target.reason && status !== "error" ? (
                <p className="text-xs text-black/50">{target.reason}</p>
              ) : null}

              {chatOpen ? (
                <div className="rounded-2xl border border-black/8 bg-white/70 p-4">
                  <div className="mb-3 flex items-center gap-2 text-xs text-black/50">
                    <Sparkles className="h-3.5 w-3.5 text-brand" />
                    Chat uses <span className="font-mono text-black/70">{job.fineTunedModel}</span>
                  </div>
                  <TestModelPanel
                    fineTunedModel={job.fineTunedModel ?? ""}
                    modelProvider={job.modelProvider}
                    baseModel={job.baseModel}
                    ollamaModelName={job.ollamaModelName}
                    ollamaRegistered={ollamaRegistered}
                  />
                </div>
              ) : null}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
