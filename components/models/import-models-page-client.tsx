"use client";

import { useEffect, useMemo, useState } from "react";
import { Boxes, Cloud, Cpu, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Input } from "@/components/ui/input";
import { LoadingState } from "@/components/ui/loading-state";
import { Textarea } from "@/components/ui/textarea";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfile, ModelProfileCategory, ModelProfilesResponse } from "@/lib/types";
import { formatDate } from "@/lib/utils";

type ImportProvider = "ollama" | "openai" | "huggingface";
type ImportMode = "ollama" | "hosted";

type ImportDraft = {
  name: string;
  provider: ImportProvider;
  model: string;
  category: ModelProfileCategory;
  description: string;
};

const importableProviders = new Set<ImportProvider>(["ollama", "openai", "huggingface"]);

const emptyOllamaDraft: ImportDraft = {
  name: "",
  provider: "ollama",
  model: "",
  category: "custom",
  description: "",
};

const emptyHostedDraft: ImportDraft = {
  name: "",
  provider: "openai",
  model: "",
  category: "custom",
  description: "",
};

const categoryOptions: Array<{ value: ModelProfileCategory; label: string }> = [
  { value: "custom", label: "Custom" },
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
  { value: "thinking", label: "Thinking" },
];

function isImportProvider(value: string): value is ImportProvider {
  return importableProviders.has(value as ImportProvider);
}

function isImportedProfile(profile: ModelProfile) {
  return !profile.id.startsWith("profile-") && isImportProvider(profile.provider);
}

function toDraft(profile: ModelProfile): ImportDraft {
  return {
    name: profile.name,
    provider: isImportProvider(profile.provider) ? profile.provider : "ollama",
    model: profile.model,
    category: profile.category,
    description: profile.description ?? "",
  };
}

function ImportForm({
  title,
  description,
  draft,
  submitLabel,
  submitting,
  onSubmit,
  onChange,
  mode,
}: {
  title: string;
  description: string;
  draft: ImportDraft;
  submitLabel: string;
  submitting: boolean;
  onSubmit: () => void;
  onChange: (updates: Partial<ImportDraft>) => void;
  mode: ImportMode;
}) {
  return (
    <Card className="space-y-5 bg-white/92">
      <div>
        <p className="font-display text-2xl">{title}</p>
        <p className="mt-2 text-sm leading-7 text-black/60">{description}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm">
          <span className="font-medium text-black/70">Display name</span>
          <Input
            value={draft.name}
            onChange={(event) => onChange({ name: event.target.value })}
            placeholder={mode === "ollama" ? "My tuned Ollama model" : "My hosted tuned model"}
          />
        </label>

        {mode === "hosted" ? (
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Hosted provider</span>
            <select
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
              value={draft.provider}
              onChange={(event) => onChange({ provider: event.target.value as ImportProvider })}
            >
              <option value="openai">OpenAI fine-tuned model id</option>
              <option value="huggingface">Hugging Face model repo id</option>
            </select>
          </label>
        ) : (
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Category</span>
            <select
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
              value={draft.category}
              onChange={(event) => onChange({ category: event.target.value as ModelProfileCategory })}
            >
              {categoryOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm md:col-span-1">
          <span className="font-medium text-black/70">{mode === "ollama" ? "Ollama model tag" : "Hosted model id"}</span>
          <Input
            value={draft.model}
            onChange={(event) => onChange({ model: event.target.value })}
            placeholder={
              mode === "ollama"
                ? "for example my-finetuned-model:latest"
                : draft.provider === "openai"
                  ? "for example ft:gpt-4.1-mini:your-org:model-id"
                  : "for example your-org/your-finetuned-model"
            }
          />
        </label>

        {mode === "hosted" ? (
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Category</span>
            <select
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
              value={draft.category}
              onChange={(event) => onChange({ category: event.target.value as ModelProfileCategory })}
            >
              {categoryOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      <label className="space-y-2 text-sm">
        <span className="font-medium text-black/70">Description</span>
        <Textarea
          value={draft.description}
          onChange={(event) => onChange({ description: event.target.value })}
          placeholder="What this tuned model is best at."
        />
      </label>

      <Button disabled={submitting || !draft.name.trim() || !draft.model.trim()} onClick={onSubmit}>
        {submitting ? "Saving..." : submitLabel}
      </Button>
    </Card>
  );
}

export function ImportModelsPageClient() {
  const [data, setData] = useState<ModelProfilesResponse | null>(null);
  const [drafts, setDrafts] = useState<Record<string, ImportDraft>>({});
  const [ollamaDraft, setOllamaDraft] = useState<ImportDraft>(emptyOllamaDraft);
  const [hostedDraft, setHostedDraft] = useState<ImportDraft>(emptyHostedDraft);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [creatingMode, setCreatingMode] = useState<ImportMode | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const payload = await pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles");
        setData(payload);
        setDrafts(
          Object.fromEntries(payload.profiles.filter(isImportedProfile).map((profile) => [profile.id, toDraft(profile)]))
        );
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Could not load imported models.");
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const importedProfiles = useMemo(
    () => (data?.profiles ?? []).filter(isImportedProfile),
    [data?.profiles]
  );

  function updateDraft(profileId: string, updates: Partial<ImportDraft>) {
    setDrafts((current) => ({
      ...current,
      [profileId]: {
        ...(current[profileId] ?? emptyOllamaDraft),
        ...updates,
      },
    }));
  }

  async function handleCreate(mode: ImportMode) {
    const draft = mode === "ollama" ? ollamaDraft : hostedDraft;
    setCreatingMode(mode);
    setMessage(null);
    setError(null);

    try {
      const created = await pythonApiFetch<ModelProfile>("/settings/model-profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      setData((current) => (current ? { ...current, profiles: [created, ...current.profiles] } : current));
      setDrafts((current) => ({ ...current, [created.id]: toDraft(created) }));
      if (mode === "ollama") {
        setOllamaDraft(emptyOllamaDraft);
      } else {
        setHostedDraft(emptyHostedDraft);
      }
      setMessage(`${created.name} is ready to use in Playground and profile selectors.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not import the model.");
    } finally {
      setCreatingMode(null);
    }
  }

  async function handleSave(profileId: string) {
    const draft = drafts[profileId];
    if (!draft) return;

    setSavingId(profileId);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfile>(`/settings/model-profiles/${profileId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      setData((current) =>
        current ? { ...current, profiles: current.profiles.map((profile) => (profile.id === profileId ? updated : profile)) } : current
      );
      setDrafts((current) => ({ ...current, [profileId]: toDraft(updated) }));
      setMessage(`Updated ${updated.name}.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not save the imported model.");
    } finally {
      setSavingId(null);
    }
  }

  async function handleDelete(profileId: string) {
    setDeletingId(profileId);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfilesResponse>(`/settings/model-profiles/${profileId}`, {
        method: "DELETE",
      });
      setData(updated);
      setDrafts(Object.fromEntries(updated.profiles.filter(isImportedProfile).map((profile) => [profile.id, toDraft(profile)])));
      setMessage("Imported model removed.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete the imported model.");
    } finally {
      setDeletingId(null);
    }
  }

  if (loading) {
    return <LoadingState message="Loading imported models..." />;
  }

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden p-0">
        <div className="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(238,247,255,0.9))] p-7 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-brand/15 bg-brand/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand">
              <Boxes className="h-3.5 w-3.5" />
              Import Existing Models
            </div>

            <h1 className="mt-5 max-w-3xl font-display text-4xl leading-tight sm:text-5xl">
              Bring already fine-tuned models into one place.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-black/62 sm:text-base">
              Save local Ollama models or hosted fine-tuned model ids as reusable profiles, then use them in Playground and anywhere else the app reads saved profiles.
            </p>
          </div>

          <div className="bg-[linear-gradient(180deg,rgba(12,29,28,0.98),rgba(25,64,57,0.94))] p-7 text-white sm:p-8">
            <p className="text-xs uppercase tracking-[0.22em] text-white/55">How to use a fine-tuned model here</p>
            <div className="mt-4 space-y-3">
              {[
                "1. Import your Ollama tag or hosted model id",
                "2. Open Playground to compare it against another saved profile",
                "3. Set it as a workspace default if you use it often",
                "4. Keep editing or deleting it from this page",
              ].map((step) => (
                <div key={step} className="rounded-[1.3rem] border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                  {step}
                </div>
              ))}
            </div>
            <p className="mt-5 text-sm leading-7 text-white/68">This page stores reusable model identifiers, not adapter file paths.</p>
          </div>
        </div>
      </Card>

      {error ? <ErrorAlert title="Could not manage imported models" description={error} /> : null}
      {message ? <Card className="border-brand/20 bg-brand/5 py-4 text-sm text-black/70">{message}</Card> : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <ImportForm
          title="Import local Ollama model"
          description="Register an Ollama tag that already exists on your machine, such as an exported fine-tuned model you can already run with `ollama run`."
          draft={ollamaDraft}
          submitLabel="Save Ollama model"
          submitting={creatingMode === "ollama"}
          onSubmit={() => void handleCreate("ollama")}
          onChange={(updates) => setOllamaDraft((current) => ({ ...current, ...updates, provider: "ollama" }))}
          mode="ollama"
        />

        <ImportForm
          title="Import hosted fine-tuned model"
          description="Register an existing OpenAI fine-tuned model id or Hugging Face model repo id so it becomes reusable inside the app."
          draft={hostedDraft}
          submitLabel="Save hosted model"
          submitting={creatingMode === "hosted"}
          onSubmit={() => void handleCreate("hosted")}
          onChange={(updates) => setHostedDraft((current) => ({ ...current, ...updates }))}
          mode="hosted"
        />
      </div>

      <Card className="space-y-5 bg-white/92">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-display text-3xl">Imported model profiles</p>
            <p className="mt-2 text-sm text-black/60">
              These imported entries are saved like normal model profiles, so they are available immediately in Playground and compatible default selectors.
            </p>
          </div>
          <div className="flex gap-2 text-xs text-black/45">
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-3 py-1">
              <Cpu className="h-3.5 w-3.5" />
              Local Ollama
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-3 py-1">
              <Cloud className="h-3.5 w-3.5" />
              Hosted ids
            </span>
          </div>
        </div>

        {importedProfiles.length === 0 ? (
          <div className="rounded-[1.5rem] border border-dashed border-black/10 bg-muted/30 px-5 py-8 text-sm text-black/55">
            No imported models yet. Add your first Ollama tag or hosted fine-tuned model id above.
          </div>
        ) : (
          <div className="space-y-4">
            {importedProfiles.map((profile) => {
              const draft = drafts[profile.id] ?? toDraft(profile);
              return (
                <div key={profile.id} className="rounded-[1.5rem] border border-black/8 bg-white px-5 py-5 shadow-sm">
                  <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
                    <label className="space-y-2 text-sm">
                      <span className="font-medium text-black/70">Display name</span>
                      <Input value={draft.name} onChange={(event) => updateDraft(profile.id, { name: event.target.value })} />
                    </label>

                    <label className="space-y-2 text-sm">
                      <span className="font-medium text-black/70">Provider</span>
                      <select
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                        value={draft.provider}
                        onChange={(event) => updateDraft(profile.id, { provider: event.target.value as ImportProvider })}
                      >
                        <option value="ollama">Ollama</option>
                        <option value="openai">OpenAI</option>
                        <option value="huggingface">Hugging Face</option>
                      </select>
                    </label>

                    <label className="space-y-2 text-sm">
                      <span className="font-medium text-black/70">Model identifier</span>
                      <Input value={draft.model} onChange={(event) => updateDraft(profile.id, { model: event.target.value })} />
                    </label>

                    <label className="space-y-2 text-sm">
                      <span className="font-medium text-black/70">Category</span>
                      <select
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                        value={draft.category}
                        onChange={(event) => updateDraft(profile.id, { category: event.target.value as ModelProfileCategory })}
                      >
                        {categoryOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <label className="mt-4 block space-y-2 text-sm">
                    <span className="font-medium text-black/70">Description</span>
                    <Textarea value={draft.description} onChange={(event) => updateDraft(profile.id, { description: event.target.value })} />
                  </label>

                  <div className="mt-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div className="text-xs text-black/45">
                      Saved {formatDate(profile.updatedAt || profile.createdAt)} • {profile.providerLabel}
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="secondary"
                        disabled={savingId === profile.id || deletingId === profile.id || !draft.name.trim() || !draft.model.trim()}
                        onClick={() => void handleSave(profile.id)}
                      >
                        {savingId === profile.id ? "Saving..." : "Save changes"}
                      </Button>
                      <Button
                        variant="ghost"
                        disabled={deletingId === profile.id || savingId === profile.id}
                        onClick={() => void handleDelete(profile.id)}
                      >
                        {deletingId === profile.id ? "Removing..." : "Delete"}
                      </Button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        {[
          {
            icon: Sparkles,
            title: "Use in Playground",
            copy: "Every imported model becomes selectable in Playground after it is saved.",
          },
          {
            icon: Cpu,
            title: "Keep local models simple",
            copy: "Use Ollama imports for models that already work with `ollama run your-model`.",
          },
          {
            icon: Cloud,
            title: "Hosted ids stay lightweight",
            copy: "OpenAI and Hugging Face imports only store the reusable model id, not training artifacts.",
          },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <Card key={item.title} className="space-y-4 bg-white/88">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <p className="font-display text-2xl">{item.title}</p>
                <p className="mt-2 text-sm leading-7 text-black/58">{item.copy}</p>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
