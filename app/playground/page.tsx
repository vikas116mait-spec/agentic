"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { PlaygroundComparison } from "@/components/playground/playground-comparison";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Textarea } from "@/components/ui/textarea";
import { findModelProfile, isRunnableProfile, modelProfileLabel } from "@/lib/model-profiles";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfilesResponse } from "@/lib/types";

const selectClassName = "rounded-2xl border border-black/10 bg-white px-4 py-3";

export default function PlaygroundPage() {
  const [prompt, setPrompt] = useState("");
  const [profilesData, setProfilesData] = useState<ModelProfilesResponse | null>(null);
  const [baseProfileId, setBaseProfileId] = useState("");
  const [compareProfileId, setCompareProfileId] = useState("");
  const [baseOutput, setBaseOutput] = useState<string | null>(null);
  const [tunedOutput, setTunedOutput] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles")
      .then((payload) => {
        const runnableProfiles = payload.profiles.filter(isRunnableProfile);
        const preferredBaseProfile = findModelProfile(runnableProfiles, payload.defaults.playgroundBaseProfileId);
        const preferredCompareProfile = findModelProfile(payload.profiles, payload.defaults.playgroundCompareProfileId);
        setProfilesData(payload);
        setBaseProfileId((current) => current || preferredBaseProfile?.id || runnableProfiles[0]?.id || payload.profiles[0]?.id || "");
        setCompareProfileId((current) => current || preferredCompareProfile?.id || "");
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  const baseProfile = useMemo(
    () => findModelProfile(profilesData?.profiles ?? [], baseProfileId),
    [baseProfileId, profilesData?.profiles]
  );
  const compareProfile = useMemo(
    () => findModelProfile(profilesData?.profiles ?? [], compareProfileId),
    [compareProfileId, profilesData?.profiles]
  );

  async function runPrompt() {
    if (!baseProfile) {
      setError("Choose a base profile before running the playground.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = await pythonApiFetch<{ baseOutput: string | null; tunedOutput: string | null }>("/playground/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt,
          baseModel: baseProfile.model,
          baseModelProvider: baseProfile.provider,
          fineTunedModel: compareProfile?.model || undefined,
          fineTunedModelProvider: compareProfile?.provider || undefined
        })
      });
      setBaseOutput(payload.baseOutput ?? null);
      setTunedOutput(payload.tunedOutput ?? null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Playground request failed.");
    }

    setLoading(false);
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Profile-driven playground</p>
          <p className="font-display text-3xl">Playground</p>
          <p className="mt-2 text-sm text-black/60">
            Compare two saved model profiles side by side using the same prompt. Need a tuned model here? Add it from
            My models, then come back and select it here.
          </p>
        </div>

        <div className="rounded-2xl border border-brand/15 bg-brand/5 p-4 text-sm text-black/70">
          Need to use a fine-tuned model that already exists?{" "}
          <Link href="/models" className="font-semibold text-brand hover:opacity-80">
            Open My models
          </Link>{" "}
          to save the model tag or hosted model id first.
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <select className={selectClassName} value={baseProfileId} onChange={(event) => setBaseProfileId(event.target.value)}>
            <option value="">Choose base profile</option>
            {(profilesData?.profiles ?? []).map((profile) => (
              <option key={profile.id} value={profile.id}>
                {modelProfileLabel(profile)}
              </option>
            ))}
          </select>
          <select className={selectClassName} value={compareProfileId} onChange={(event) => setCompareProfileId(event.target.value)}>
            <option value="">No comparison profile</option>
            {(profilesData?.profiles ?? []).map((profile) => (
              <option key={profile.id} value={profile.id}>
                {modelProfileLabel(profile)}
              </option>
            ))}
          </select>
        </div>

        {baseProfile ? (
          <div className="rounded-2xl bg-white p-4 text-sm text-black/65">
            Base profile: {modelProfileLabel(baseProfile)}{baseProfile.providerConfigured ? "" : " | provider not configured"}
          </div>
        ) : null}

        {compareProfile ? (
          <div className="rounded-2xl bg-white p-4 text-sm text-black/65">
            Comparison profile: {modelProfileLabel(compareProfile)}
            {compareProfile.providerConfigured ? "" : " | provider not configured"}
          </div>
        ) : null}

        <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Enter the prompt you want to test..." />

        {error ? <ErrorAlert title="Could not run prompt" description={error} /> : null}

        <Button
          disabled={
            loading ||
            !prompt.trim() ||
            !baseProfile ||
            !isRunnableProfile(baseProfile) ||
            (compareProfile ? !isRunnableProfile(compareProfile) : false)
          }
          onClick={runPrompt}
        >
          {loading ? "Running..." : "Run comparison"}
        </Button>
      </Card>

      <PlaygroundComparison baseOutput={baseOutput} tunedOutput={tunedOutput} />
    </div>
  );
}
