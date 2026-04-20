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
import { cn } from "@/lib/utils";
import { formatDate } from "@/lib/utils";

type ProviderKey = "ollama" | "openai" | "huggingface" | "local" | "groq" | "gemini" | "cerebras" | "together";

type ProfileDraft = {
  name: string;
  provider: ProviderKey;
  model: string;
  category: ModelProfileCategory;
  description: string;
};

type ModelOption = {
  value: string;
  label: string;
  hint: string;
};

type SettingsTab = "profiles" | "defaults" | "providers";

const MODEL_OPTIONS: Record<ProviderKey, ModelOption[]> = {
  ollama: [
    { value: "qwen3:8b", label: "Qwen 3 8B", hint: "Balanced local default" },
    { value: "qwen3:4b", label: "Qwen 3 4B", hint: "Fast small local model" },
    { value: "llama3.1:8b", label: "Llama 3.1 8B", hint: "General-purpose local model" },
    { value: "deepseek-r1:8b", label: "DeepSeek R1 8B", hint: "Reasoning-heavy local model" }
  ],
  openai: [
    { value: "gpt-4.1-mini-2025-04-14", label: "GPT-4.1 Mini", hint: "Paid hosted fine-tuning model" },
    { value: "gpt-4.1-2025-04-14", label: "GPT-4.1", hint: "Paid hosted higher-quality model" },
    { value: "gpt-5.4-mini", label: "GPT-5.4 Mini", hint: "Paid hosted reasoning model" }
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
  ],
  groq: [
    { value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B Versatile", hint: "Best quality free Groq model" },
    { value: "mixtral-8x7b-32768", label: "Mixtral 8x7B 32k", hint: "Fast MoE model on Groq" },
    { value: "gemma2-9b-it", label: "Gemma 2 9B IT", hint: "Smallest free Groq model" },
    { value: "llama-3.2-3b-preview", label: "Llama 3.2 3B Preview", hint: "Ultra-fast tiny Llama on Groq" }
  ],
  gemini: [
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash", hint: "Latest free Gemini model" },
    { value: "gemini-1.5-flash", label: "Gemini 1.5 Flash", hint: "Reliable free Gemini model" },
    { value: "gemini-1.5-flash-8b", label: "Gemini 1.5 Flash 8B", hint: "Smallest free Gemini model" }
  ],
  cerebras: [
    { value: "llama-4-scout-17b-16e-instruct", label: "Llama 4 Scout 17B", hint: "Meta Llama 4 on Cerebras hardware" },
    { value: "llama3.1-70b", label: "Llama 3.1 70B", hint: "Ultra-fast 70B on wafer-scale chip" }
  ],
  together: [
    { value: "meta-llama/Meta-Llama-3.1-70B-Instruct", label: "Llama 3.1 70B Instruct", hint: "Strong open-source 70B via Together" },
    { value: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen 2.5 72B Instruct", hint: "Top open-source 72B via Together" },
    { value: "mistralai/Mixtral-8x7B-Instruct-v0.1", label: "Mixtral 8x7B Instruct", hint: "Popular MoE model via Together" }
  ]
};

const CUSTOM_MODEL_VALUE = "__custom__";

function defaultModelForProvider(provider: ProviderKey) {
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
    provider: profile.provider as ProviderKey,
    model: profile.model,
    category: profile.category,
    description: profile.description ?? ""
  };
}

function modelSelectValue(provider: ProviderKey, model: string) {
  return MODEL_OPTIONS[provider]?.some((option) => option.value === model) ? model : CUSTOM_MODEL_VALUE;
}

function modelOptionsForProvider(provider: ProviderKey, model: string) {
  const options = MODEL_OPTIONS[provider] ?? [];
  if (!model || options.some((option) => option.value === model)) {
    return options;
  }
  return [{ value: model, label: `Custom | ${model}`, hint: "Saved custom model id" }, ...options];
}

type ModelFieldProps = {
  provider: ProviderKey;
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

function ProviderSelect({
  value,
  onChange
}: {
  value: ProviderKey;
  onChange: (provider: ProviderKey) => void;
}) {
  return (
    <select
      className={selectClassName}
      value={value}
      onChange={(event) => onChange(event.target.value as ProviderKey)}
    >
      <option value="ollama">Ollama</option>
      <option value="openai">OpenAI</option>
      <option value="huggingface">Hugging Face Jobs</option>
      <option value="local">Local GPU QLoRA</option>
      <option value="groq">Groq (Free tier)</option>
      <option value="gemini">Google Gemini (Free tier)</option>
      <option value="cerebras">Cerebras (Free tier)</option>
      <option value="together">Together AI ($25 credits)</option>
    </select>
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
  const [tab, setTab] = useState<SettingsTab>("profiles");

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

  function updateDraftProvider(profileId: string, provider: ProviderKey) {
    updateDraft(profileId, {
      provider,
      model: defaultModelForProvider(provider)
    });
  }

  function updateNewProfileProvider(provider: ProviderKey) {
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newProfile)
      });
      setData((current) =>
        current ? { ...current, profiles: [created, ...current.profiles] } : current
      );
      setDrafts((current) => ({ ...current, [created.id]: toProfileDraft(created) }));
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
    if (!draft) return;

    setSavingId(profileId);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfile>(`/settings/model-profiles/${profileId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft)
      });
      setData((current) =>
        current
          ? { ...current, profiles: current.profiles.map((p) => (p.id === profileId ? updated : p)) }
          : current
      );
      setDrafts((current) => ({ ...current, [profileId]: toProfileDraft(updated) }));
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
      setDrafts(Object.fromEntries(updated.profiles.map((p) => [p.id, toProfileDraft(p)])));
      setMessage("Model profile deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete the profile.");
    } finally {
      setDeletingId(null);
    }
  }

  async function handleSaveDefaults() {
    if (!data) return;

    setSavingDefaults(true);
    setMessage(null);
    setError(null);

    try {
      const updated = await pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles/defaults", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data.defaults)
      });
      setData(updated);
      setDrafts(Object.fromEntries(updated.profiles.map((p) => [p.id, toProfileDraft(p)])));
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

  const tabs: { key: SettingsTab; label: string }[] = [
    { key: "profiles", label: "Model Profiles" },
    { key: "defaults", label: "Workspace Defaults" },
    { key: "providers", label: "Providers" }
  ];

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div>
          <p className="font-display text-3xl">Models</p>
          <p className="mt-3 text-sm text-black/60">
            Manage model profiles, workspace defaults, and connected providers. For unpaid experiments, prefer Local GPU
            QLoRA or the free-tier providers (Groq, Gemini, Cerebras, Together AI).
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 rounded-2xl border border-black/10 bg-black/3 p-1 w-fit">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setTab(key)}
              className={cn(
                "rounded-xl px-5 py-2 text-sm font-medium transition",
                tab === key ? "bg-white text-black shadow-sm" : "text-black/50 hover:text-black/80"
              )}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Feedback messages — visible regardless of active tab */}
        {message ? <p className="text-sm text-black/70">{message}</p> : null}
        {error ? <ErrorAlert title="Settings action failed" description={error} /> : null}
      </Card>

      {/* Providers tab */}
      {tab === "providers" && (
        <Card className="space-y-4">
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Connected providers</p>
          <div className="grid gap-3 md:grid-cols-2">
            {data.providers.map((provider) => (
              <div key={provider.provider} className="rounded-2xl bg-white p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">{provider.label}</p>
                <p className={cn("mt-2 text-sm font-semibold", provider.configured ? "text-green-700" : "text-amber-600")}>
                  {provider.configured ? "Configured" : "Needs setup"}
                </p>
                <p className="mt-1 text-xs text-black/55">{provider.baseUrl ?? "Default API endpoint"}</p>
                <p className="mt-2 text-xs text-black/45">
                  {provider.supportsInference ? "Inference" : "Training only"} |{" "}
                  {provider.supportsFineTuning ? "Fine-tuning enabled" : "No fine-tuning"}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Workspace defaults tab */}
      {tab === "defaults" && (
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
      )}

      {/* Model profiles tab */}
      {tab === "profiles" && (
        <>
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
              <ProviderSelect value={newProfile.provider} onChange={updateNewProfileProvider} />
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
                    <div className={cn(
                      "rounded-2xl px-4 py-3 text-xs uppercase tracking-[0.2em]",
                      profile.providerConfigured ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-600"
                    )}>
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
                    <ProviderSelect
                      value={draft.provider}
                      onChange={(provider) => updateDraftProvider(profile.id, provider)}
                    />
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
                    <Button
                      variant="danger"
                      onClick={() => void handleDeleteProfile(profile.id)}
                      disabled={deletingId === profile.id}
                    >
                      {deletingId === profile.id ? "Deleting..." : "Delete profile"}
                    </Button>
                  </div>
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
