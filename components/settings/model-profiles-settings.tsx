"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Input } from "@/components/ui/input";
import { LoadingState } from "@/components/ui/loading-state";
import { Textarea } from "@/components/ui/textarea";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfile, ModelProfileCategory, ModelProfilesResponse } from "@/lib/types";
import { formatDate } from "@/lib/utils";

type ProfileDraft = {
  name: string;
  provider: "ollama" | "openai" | "huggingface" | "local";
  model: string;
  category: ModelProfileCategory;
  description: string;
};

type ModelOption = {
  value: string;
  label: string;
  hint: string;
};

const MODEL_OPTIONS: Record<ProfileDraft["provider"], ModelOption[]> = {
  ollama: [
    { value: "qwen3:8b", label: "Qwen 3 8B", hint: "Balanced local default" },
    { value: "qwen3:4b", label: "Qwen 3 4B", hint: "Fast small local model" },
    {
      value: "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
      label: "HF Llama 3.2 3B Instruct",
      hint: "Pulled from Hugging Face through Ollama"
    },
    { value: "llama3.1:8b", label: "Llama 3.1 8B", hint: "General-purpose local model" },
    { value: "deepseek-r1:8b", label: "DeepSeek R1 8B", hint: "Reasoning-heavy local model" }
  ],
  openai: [
    { value: "gpt-4.1-mini-2025-04-14", label: "GPT-4.1 Mini", hint: "Managed fine-tuning base model" },
    { value: "gpt-4.1-2025-04-14", label: "GPT-4.1", hint: "Higher-quality hosted model" },
    { value: "gpt-5.4-mini", label: "GPT-5.4 Mini", hint: "Thinking profile default" }
  ],
  huggingface: [
    { value: "Qwen/Qwen2.5-0.5B-Instruct", label: "Qwen 2.5 0.5B Instruct", hint: "Tiny free smoke-test model" },
    { value: "Qwen/Qwen2.5-1.5B-Instruct", label: "Qwen 2.5 1.5B Instruct", hint: "Smallest public starter model" },
    { value: "Qwen/Qwen2.5-3B-Instruct", label: "Qwen 2.5 3B Instruct", hint: "Best first open-source fine-tune" },
    { value: "microsoft/Phi-3.5-mini-instruct", label: "Phi 3.5 Mini Instruct", hint: "Compact Microsoft instruct model" },
    { value: "mistralai/Mistral-7B-Instruct-v0.3", label: "Mistral 7B Instruct v0.3", hint: "Popular open instruct model" },
    { value: "Qwen/Qwen2.5-7B-Instruct", label: "Qwen 2.5 7B Instruct", hint: "Stronger open-source cloud fine-tune" }
  ],
  local: [
    { value: "Qwen/Qwen2.5-0.5B-Instruct", label: "Qwen 2.5 0.5B Instruct", hint: "Fastest free smoke-test model" },
    { value: "Qwen/Qwen2.5-1.5B-Instruct", label: "Qwen 2.5 1.5B Instruct", hint: "Fastest free local starter" },
    { value: "Qwen/Qwen2.5-3B-Instruct", label: "Qwen 2.5 3B Instruct", hint: "Recommended first local fine-tune" },
    { value: "microsoft/Phi-3.5-mini-instruct", label: "Phi 3.5 Mini Instruct", hint: "Compact model with good quality per GPU" },
    { value: "mistralai/Mistral-7B-Instruct-v0.3", label: "Mistral 7B Instruct v0.3", hint: "Strong open instruct model if you want another family" },
    { value: "Qwen/Qwen2.5-7B-Instruct", label: "Qwen 2.5 7B Instruct", hint: "Larger free model if you want stronger quality" }
  ]
};

const CUSTOM_MODEL_VALUE = "__custom__";

function defaultModelForProvider(provider: ProfileDraft["provider"]) {
  return MODEL_OPTIONS[provider][0]?.value ?? "";
}

const emptyDraft: ProfileDraft = {
  name: "",
  provider: "ollama",
  model: defaultModelForProvider("ollama"),
  category: "custom",
  description: ""
};

const selectClassName = "w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm";

function toProfileDraft(profile: ModelProfile): ProfileDraft {
  return {
    name: profile.name,
    provider: profile.provider,
    model: profile.model,
    category: profile.category,
    description: profile.description ?? ""
  };
}

function modelSelectValue(provider: ProfileDraft["provider"], model: string) {
  return MODEL_OPTIONS[provider].some((option) => option.value === model) ? model : CUSTOM_MODEL_VALUE;
}

function modelOptionsForProvider(provider: ProfileDraft["provider"], model: string) {
  const options = MODEL_OPTIONS[provider];
  if (!model || options.some((option) => option.value === model)) {
    return options;
  }
  return [{ value: model, label: `Custom | ${model}`, hint: "Saved custom model id" }, ...options];
}

type ModelFieldProps = {
  provider: ProfileDraft["provider"];
  value: string;
  onChange: (nextValue: string) => void;
  label: string;
};

function ModelField({ provider, value, onChange, label }: ModelFieldProps) {
  const selectedValue = modelSelectValue(provider, value);
  const options = modelOptionsForProvider(provider, value);

  return (
    <label className="space-y-2 text-sm">
      <span className="font-medium text-black/70">{label}</span>
      <select
        className={selectClassName}
        value={selectedValue}
        onChange={(event) => onChange(event.target.value === CUSTOM_MODEL_VALUE ? "" : event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label} | {option.hint}
          </option>
        ))}
        <option value={CUSTOM_MODEL_VALUE}>Custom model id...</option>
      </select>

      {selectedValue === CUSTOM_MODEL_VALUE ? (
        <Input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Enter a custom model id"
        />
      ) : null}
    </label>
  );
}

export function ModelProfilesSettings() {
  const [data, setData] = useState<ModelProfilesResponse | null>(null);
  const [drafts, setDrafts] = useState<Record<string, ProfileDraft>>({});
  const [newProfile, setNewProfile] = useState<ProfileDraft>(emptyDraft);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [savingDefaults, setSavingDefaults] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const payload = await pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles");
        setData(payload);
        setDrafts(
          Object.fromEntries(payload.profiles.map((profile) => [profile.id, toProfileDraft(profile)]))
        );
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Could not load model settings.");
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  function updateDraft(profileId: string, updates: Partial<ProfileDraft>) {
    setDrafts((current) => ({
      ...current,
      [profileId]: {
        ...(current[profileId] ?? emptyDraft),
        ...updates
      }
    }));
  }

  function updateDraftProvider(profileId: string, provider: ProfileDraft["provider"]) {
    updateDraft(profileId, {
      provider,
      model: defaultModelForProvider(provider)
    });
  }

  function updateNewProfileProvider(provider: ProfileDraft["provider"]) {
    setNewProfile((current) => ({
      ...current,
      provider,
      model: defaultModelForProvider(provider)
    }));
  }

  function updateDefaults(updates: Partial<ModelProfilesResponse["defaults"]>) {
    setData((current) => (current ? { ...current, defaults: { ...current.defaults, ...updates } } : current));
  }

  async function handleCreateProfile() {
    setCreating(true);
    setMessage(null);
    setError(null);

    try {
      const created = await pythonApiFetch<ModelProfile>("/settings/model-profiles", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newProfile)
      });
      setData((current) =>
        current
          ? {
              ...current,
              profiles: [created, ...current.profiles]
            }
          : current
      );
      setDrafts((current) => ({
        ...current,
        [created.id]: toProfileDraft(created)
      }));
      setNewProfile(emptyDraft);
      setMessage("Model profile created.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not create the profile.");
    } finally {
      setCreating(false);
    }
  }

  async function handleSaveProfile(profileId: string) {
    const draft = drafts[profileId];
    if (!draft) {
      return;
    }

    setSavingId(profileId);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfile>(`/settings/model-profiles/${profileId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(draft)
      });
      setData((current) =>
        current
          ? {
              ...current,
              profiles: current.profiles.map((profile) => (profile.id === profileId ? updated : profile))
            }
          : current
      );
      setDrafts((current) => ({
        ...current,
        [profileId]: toProfileDraft(updated)
      }));
      setMessage(`Saved ${updated.name}.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not save the profile.");
    } finally {
      setSavingId(null);
    }
  }

  async function handleDeleteProfile(profileId: string) {
    setDeletingId(profileId);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfilesResponse>(`/settings/model-profiles/${profileId}`, {
        method: "DELETE"
      });
      setData(updated);
      setDrafts(Object.fromEntries(updated.profiles.map((profile) => [profile.id, toProfileDraft(profile)])));
      setMessage("Model profile deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete the profile.");
    } finally {
      setDeletingId(null);
    }
  }

  async function handleSaveDefaults() {
    if (!data) {
      return;
    }

    setSavingDefaults(true);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles/defaults", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data.defaults)
      });
      setData(updated);
      setDrafts(Object.fromEntries(updated.profiles.map((profile) => [profile.id, toProfileDraft(profile)])));
      setMessage("Workspace defaults updated.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not save the defaults.");
    } finally {
      setSavingDefaults(false);
    }
  }

  if (loading) {
    return <LoadingState label="Loading model settings..." />;
  }

  if (error && !data) {
    return <ErrorAlert title="Could not load model settings" description={error} />;
  }

  if (!data) {
    return <ErrorAlert title="Could not load model settings" description="No settings payload was returned." />;
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div>
          <p className="font-display text-3xl">Models</p>
          <p className="mt-3 text-sm text-black/60">
            This page is only for changing the base models or providers behind training. If you just want to fine-tune
            quickly, the default profiles are already enough. For unpaid experiments, prefer the Local GPU QLoRA models
            first.
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {data.providers.map((provider) => (
            <div key={provider.provider} className="rounded-2xl bg-white p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">{provider.label}</p>
              <p className="mt-2 text-sm font-semibold">{provider.configured ? "Configured" : "Needs setup"}</p>
              <p className="mt-1 text-xs text-black/55">{provider.baseUrl ?? "Default API endpoint"}</p>
              <p className="mt-2 text-xs text-black/45">
                {provider.supportsInference ? "Inference" : "Training only"} |{" "}
                {provider.supportsFineTuning ? "Fine-tuning enabled" : "No fine-tuning"}
              </p>
            </div>
          ))}
        </div>

        {message ? <p className="text-sm text-black/70">{message}</p> : null}
        {error ? <ErrorAlert title="Settings action failed" description={error} /> : null}
      </Card>

      <Card className="space-y-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Workspace defaults</p>
          <h2 className="mt-2 font-display text-3xl">Choose the default models</h2>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Playground base profile</span>
            <select
              className={selectClassName}
              value={data.defaults.playgroundBaseProfileId ?? ""}
              onChange={(event) => updateDefaults({ playgroundBaseProfileId: event.target.value || null })}
            >
              <option value="">None</option>
              {data.profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} | {profile.providerLabel} | {profile.model}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Playground comparison profile</span>
            <select
              className={selectClassName}
              value={data.defaults.playgroundCompareProfileId ?? ""}
              onChange={(event) => updateDefaults({ playgroundCompareProfileId: event.target.value || null })}
            >
              <option value="">None</option>
              {data.profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} | {profile.providerLabel} | {profile.model}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Agent base profile</span>
            <select
              className={selectClassName}
              value={data.defaults.agentBaseProfileId ?? ""}
              onChange={(event) => updateDefaults({ agentBaseProfileId: event.target.value || null })}
            >
              <option value="">None</option>
              {data.profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} | {profile.providerLabel} | {profile.model}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Agent reasoning profile</span>
            <select
              className={selectClassName}
              value={data.defaults.agentModelProfileId ?? ""}
              onChange={(event) => updateDefaults({ agentModelProfileId: event.target.value || null })}
            >
              <option value="">None</option>
              {data.profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} | {profile.providerLabel} | {profile.model}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2 text-sm md:col-span-2">
            <span className="font-medium text-black/70">Default training profile</span>
            <select
              className={selectClassName}
              value={data.defaults.jobBaseProfileId ?? ""}
              onChange={(event) => updateDefaults({ jobBaseProfileId: event.target.value || null })}
            >
              <option value="">None</option>
              {data.profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} | {profile.providerLabel} | {profile.model}
                </option>
              ))}
            </select>
          </label>
        </div>

        <Button onClick={handleSaveDefaults} disabled={savingDefaults}>
          {savingDefaults ? "Saving..." : "Save workspace defaults"}
        </Button>
      </Card>

      <Card className="space-y-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">New profile</p>
          <h2 className="mt-2 font-display text-3xl">Add another model</h2>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm">
            <span className="font-medium text-black/70">Profile name</span>
            <Input
              value={newProfile.name}
              onChange={(event) => setNewProfile((current) => ({ ...current, name: event.target.value }))}
              placeholder="Profile name"
            />
          </label>
          <ModelField
            provider={newProfile.provider}
            value={newProfile.model}
            onChange={(model) => setNewProfile((current) => ({ ...current, model }))}
            label="Model"
          />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <select
            className={selectClassName}
            value={newProfile.provider}
            onChange={(event) => updateNewProfileProvider(event.target.value as ProfileDraft["provider"])}
          >
            <option value="ollama">Ollama</option>
            <option value="openai">OpenAI</option>
            <option value="huggingface">Hugging Face Jobs</option>
            <option value="local">Local GPU QLoRA</option>
          </select>
          <select
            className={selectClassName}
            value={newProfile.category}
            onChange={(event) =>
              setNewProfile((current) => ({ ...current, category: event.target.value as ModelProfileCategory }))
            }
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
            <option value="thinking">Thinking</option>
            <option value="custom">Custom</option>
          </select>
        </div>

        <Textarea
          className="min-h-24"
          value={newProfile.description}
          onChange={(event) => setNewProfile((current) => ({ ...current, description: event.target.value }))}
          placeholder="Optional note about when this profile should be used"
        />

        <Button onClick={handleCreateProfile} disabled={creating}>
          {creating ? "Creating..." : "Create profile"}
        </Button>
      </Card>

      <div className="space-y-4">
        {data.profiles.map((profile) => {
          const draft = drafts[profile.id] ?? toProfileDraft(profile);

          return (
            <Card key={profile.id} className="space-y-4">
              <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="font-display text-2xl">{profile.name}</p>
                  <p className="mt-2 text-sm text-black/60">
                    {profile.providerLabel} | {profile.model} | updated {formatDate(profile.updatedAt)}
                  </p>
                </div>
                <div className="rounded-2xl bg-white px-4 py-3 text-xs uppercase tracking-[0.2em] text-black/45">
                  {profile.providerConfigured ? "Ready to use" : "Needs provider setup"}
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-2 text-sm">
                  <span className="font-medium text-black/70">Profile name</span>
                  <Input
                    value={draft.name}
                    onChange={(event) => updateDraft(profile.id, { name: event.target.value })}
                    placeholder="Profile name"
                  />
                </label>
                <ModelField
                  provider={draft.provider}
                  value={draft.model}
                  onChange={(model) => updateDraft(profile.id, { model })}
                  label="Model"
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <select
                  className={selectClassName}
                  value={draft.provider}
                  onChange={(event) => updateDraftProvider(profile.id, event.target.value as ProfileDraft["provider"])}
                >
                  <option value="ollama">Ollama</option>
                  <option value="openai">OpenAI</option>
                  <option value="huggingface">Hugging Face Jobs</option>
                  <option value="local">Local GPU QLoRA</option>
                </select>
                <select
                  className={selectClassName}
                  value={draft.category}
                  onChange={(event) =>
                    updateDraft(profile.id, { category: event.target.value as ModelProfileCategory })
                  }
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                  <option value="thinking">Thinking</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              <Textarea
                className="min-h-24"
                value={draft.description}
                onChange={(event) => updateDraft(profile.id, { description: event.target.value })}
                placeholder="Optional note about when this profile should be used"
              />

              <div className="flex flex-wrap gap-3">
                <Button onClick={() => void handleSaveProfile(profile.id)} disabled={savingId === profile.id}>
                  {savingId === profile.id ? "Saving..." : "Save profile"}
                </Button>
                <Button variant="danger" onClick={() => void handleDeleteProfile(profile.id)} disabled={deletingId === profile.id}>
                  {deletingId === profile.id ? "Deleting..." : "Delete profile"}
                </Button>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
