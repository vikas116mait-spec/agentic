"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Boxes, Cloud, Cpu, PencilLine, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Input } from "@/components/ui/input";
import { LoadingState } from "@/components/ui/loading-state";
import { Textarea } from "@/components/ui/textarea";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfile, ModelProfileCategory, ModelProfilesResponse } from "@/lib/types";
import { cn, formatDate } from "@/lib/utils";

type ImportProvider = "ollama" | "openai" | "huggingface";

type ImportDraft = {
  name: string;
  provider: ImportProvider;
  model: string;
  category: ModelProfileCategory;
  description: string;
};

const importableProviders = new Set<ImportProvider>(["ollama", "openai", "huggingface"]);

const emptyDraft: ImportDraft = {
  name: "",
  provider: "ollama",
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

const providerOptions: Array<{
  value: ImportProvider;
  label: string;
  hint: string;
  example: string;
  icon: typeof Cpu;
}> = [
  {
    value: "ollama",
    label: "Ollama",
    hint: "Use this for a local model that already works with `ollama run`.",
    example: "my-finetuned-model:latest",
    icon: Cpu,
  },
  {
    value: "openai",
    label: "OpenAI",
    hint: "Use this for an existing OpenAI fine-tuned model id.",
    example: "ft:gpt-4.1-mini:your-org:model-id",
    icon: Sparkles,
  },
  {
    value: "huggingface",
    label: "Hugging Face",
    hint: "Use this for a hosted Hugging Face model repo id.",
    example: "your-org/your-finetuned-model",
    icon: Cloud,
  },
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

function providerLabel(provider: ImportProvider) {
  return providerOptions.find((option) => option.value === provider)?.label ?? provider;
}

function providerHint(provider: ImportProvider) {
  return providerOptions.find((option) => option.value === provider)?.hint ?? "";
}

function providerExample(provider: ImportProvider) {
  return providerOptions.find((option) => option.value === provider)?.example ?? "";
}

function summaryText(profile: ModelProfile) {
  return `${profile.providerLabel} • ${profile.model}`;
}

export function ImportModelsPageClient() {
  const [data, setData] = useState<ModelProfilesResponse | null>(null);
  const [drafts, setDrafts] = useState<Record<string, ImportDraft>>({});
  const [createDraft, setCreateDraft] = useState<ImportDraft>(emptyDraft);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const payload = await pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles");
        setData(payload);
        setDrafts(
          Object.fromEntries(payload.profiles.filter(isImportedProfile).map((profile) => [profile.id, toDraft(profile)]))
        );
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Could not load saved models.");
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const importedProfiles = useMemo(() => (data?.profiles ?? []).filter(isImportedProfile), [data?.profiles]);
  const createProviderMeta = providerOptions.find((option) => option.value === createDraft.provider) ?? providerOptions[0];

  function updateDraft(profileId: string, updates: Partial<ImportDraft>) {
    setDrafts((current) => ({
      ...current,
      [profileId]: {
        ...(current[profileId] ?? emptyDraft),
        ...updates,
      },
    }));
  }

  function resetMessages() {
    setMessage(null);
    setError(null);
  }

  async function handleCreate() {
    setCreating(true);
    resetMessages();

    try {
      const created = await pythonApiFetch<ModelProfile>("/settings/model-profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(createDraft),
      });
      setData((current) => (current ? { ...current, profiles: [created, ...current.profiles] } : current));
      setDrafts((current) => ({ ...current, [created.id]: toDraft(created) }));
      setCreateDraft((current) => ({ ...emptyDraft, provider: current.provider }));
      setMessage(`${created.name} is now ready in Playground and other profile pickers.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not save the model.");
    } finally {
      setCreating(false);
    }
  }

  async function handleSave(profileId: string) {
    const draft = drafts[profileId];
    if (!draft) return;

    setSavingId(profileId);
    resetMessages();

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
      setEditingId(null);
      setMessage(`Updated ${updated.name}.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not update the model.");
    } finally {
      setSavingId(null);
    }
  }

  async function handleDelete(profileId: string) {
    setDeletingId(profileId);
    resetMessages();

    try {
      const updated = await pythonApiFetch<ModelProfilesResponse>(`/settings/model-profiles/${profileId}`, {
        method: "DELETE",
      });
      setData(updated);
      setDrafts(Object.fromEntries(updated.profiles.filter(isImportedProfile).map((profile) => [profile.id, toDraft(profile)])));
      setEditingId((current) => (current === profileId ? null : current));
      setMessage("Model removed.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete the model.");
    } finally {
      setDeletingId(null);
    }
  }

  if (loading) {
    return <LoadingState message="Loading saved models..." />;
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-6 bg-white/95">
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-brand/15 bg-brand/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand">
            <Boxes className="h-3.5 w-3.5" />
            My models
          </div>
          <div>
            <h1 className="font-display text-4xl leading-tight sm:text-5xl">Add a fine-tuned model in one step.</h1>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-black/62 sm:text-base">
              Name it, paste the model tag or model id, and save it. The model becomes available in Playground and anywhere else
              the app uses saved profiles.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {providerOptions.map((option) => {
            const Icon = option.icon;
            const selected = createDraft.provider === option.value;
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => setCreateDraft((current) => ({ ...current, provider: option.value }))}
                className={cn(
                  "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition",
                  selected ? "border-brand bg-brand text-brand-foreground" : "border-black/10 bg-white text-black/70 hover:bg-black/5"
                )}
              >
                <Icon className="h-4 w-4" />
                {option.label}
              </button>
            );
          })}
        </div>

        <div className="rounded-[1.5rem] border border-black/8 bg-muted/20 p-4 text-sm text-black/65">
          <p>{createProviderMeta.hint}</p>
          <p className="mt-2 text-black/50">Example: `{createProviderMeta.example}`</p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Display name</span>
            <Input
              value={createDraft.name}
              onChange={(event) => setCreateDraft((current) => ({ ...current, name: event.target.value }))}
              placeholder="My fine-tuned model"
            />
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">
              {createDraft.provider === "ollama" ? "Model tag" : "Model id"}
            </span>
            <Input
              value={createDraft.model}
              onChange={(event) => setCreateDraft((current) => ({ ...current, model: event.target.value }))}
              placeholder={providerExample(createDraft.provider)}
            />
          </label>
        </div>

        <details className="rounded-[1.5rem] border border-black/8 bg-white px-5 py-4">
          <summary className="cursor-pointer list-none text-sm font-medium text-black/70">Advanced options</summary>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <label className="space-y-2 text-sm">
              <span className="font-medium text-black/70">Category</span>
              <select
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                value={createDraft.category}
                onChange={(event) => setCreateDraft((current) => ({ ...current, category: event.target.value as ModelProfileCategory }))}
              >
                {categoryOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-2 text-sm md:col-span-2">
              <span className="font-medium text-black/70">Description</span>
              <Textarea
                value={createDraft.description}
                onChange={(event) => setCreateDraft((current) => ({ ...current, description: event.target.value }))}
                placeholder="Optional note about what this model is best at."
              />
            </label>
          </div>
        </details>

        <div className="flex flex-wrap items-center gap-3">
          <Button disabled={creating || !createDraft.name.trim() || !createDraft.model.trim()} onClick={() => void handleCreate()}>
            {creating ? "Saving..." : "Save model"}
          </Button>
          <Link href="/playground" className="text-sm font-medium text-brand hover:opacity-80">
            Open Playground
          </Link>
        </div>
      </Card>

      {error ? <ErrorAlert title="Could not manage models" description={error} /> : null}
      {message ? <Card className="border-brand/20 bg-brand/5 py-4 text-sm text-black/70">{message}</Card> : null}

      <Card className="space-y-5 bg-white/92">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-display text-3xl">Saved models</p>
            <p className="mt-2 text-sm text-black/60">
              These entries are reusable profile shortcuts. Save once, then pick them anywhere the app asks for a saved model.
            </p>
          </div>
          <div className="text-sm text-black/45">{importedProfiles.length} saved</div>
        </div>

        {importedProfiles.length === 0 ? (
          <div className="rounded-[1.5rem] border border-dashed border-black/10 bg-muted/30 px-5 py-8 text-sm text-black/55">
            No saved models yet. Add your first model above, then use it in Playground.
          </div>
        ) : (
          <div className="space-y-3">
            {importedProfiles.map((profile) => {
              const draft = drafts[profile.id] ?? toDraft(profile);
              const isEditing = editingId === profile.id;
              return (
                <div key={profile.id} className="rounded-[1.5rem] border border-black/8 bg-white px-5 py-5 shadow-sm">
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-display text-2xl">{profile.name}</p>
                        <span className="rounded-full bg-muted px-3 py-1 text-xs text-black/55">{profile.providerLabel}</span>
                      </div>
                      <p className="text-sm text-black/62">{summaryText(profile)}</p>
                      <p className="text-xs text-black/45">Saved {formatDate(profile.updatedAt || profile.createdAt)}</p>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={savingId === profile.id || deletingId === profile.id}
                        onClick={() => {
                          setEditingId((current) => (current === profile.id ? null : profile.id));
                          setDrafts((current) => ({ ...current, [profile.id]: toDraft(profile) }));
                        }}
                      >
                        <PencilLine className="mr-1 h-3.5 w-3.5" />
                        {isEditing ? "Close" : "Edit"}
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={deletingId === profile.id || savingId === profile.id}
                        onClick={() => void handleDelete(profile.id)}
                      >
                        {deletingId === profile.id ? "Removing..." : "Delete"}
                      </Button>
                    </div>
                  </div>

                  {isEditing ? (
                    <div className="mt-5 space-y-4 border-t border-black/8 pt-5">
                      <div className="grid gap-4 md:grid-cols-2">
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
                            {providerOptions.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>

                        <label className="space-y-2 text-sm md:col-span-2">
                          <span className="font-medium text-black/70">Model identifier</span>
                          <Input value={draft.model} onChange={(event) => updateDraft(profile.id, { model: event.target.value })} />
                        </label>
                      </div>

                      <details className="rounded-[1.25rem] border border-black/8 bg-muted/15 px-4 py-3">
                        <summary className="cursor-pointer list-none text-sm font-medium text-black/70">Advanced options</summary>
                        <div className="mt-4 grid gap-4 md:grid-cols-2">
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

                          <label className="space-y-2 text-sm md:col-span-2">
                            <span className="font-medium text-black/70">Description</span>
                            <Textarea
                              value={draft.description}
                              onChange={(event) => updateDraft(profile.id, { description: event.target.value })}
                            />
                          </label>
                        </div>
                      </details>

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
                          disabled={savingId === profile.id || deletingId === profile.id}
                          onClick={() => {
                            setDrafts((current) => ({ ...current, [profile.id]: toDraft(profile) }));
                            setEditingId(null);
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}
