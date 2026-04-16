"use client";

import { useEffect, useState } from "react";
import { PlaygroundComparison } from "@/components/playground/playground-comparison";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Textarea } from "@/components/ui/textarea";
import { pythonApiFetch } from "@/lib/python-api";

type RuntimeStatus = {
  modelProviderLabel: string;
  defaultBaseModel: string;
};

export default function PlaygroundPage() {
  const [prompt, setPrompt] = useState("");
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [baseModel, setBaseModel] = useState("");
  const [fineTunedModel, setFineTunedModel] = useState("");
  const [baseOutput, setBaseOutput] = useState<string | null>(null);
  const [tunedOutput, setTunedOutput] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<RuntimeStatus>("/agent/runtime")
      .then((payload) => {
        setRuntime(payload);
        setBaseModel((current) => current || payload.defaultBaseModel);
      })
      .catch(() => undefined);
  }, []);

  async function runPrompt() {
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
          baseModel,
          fineTunedModel: fineTunedModel || undefined
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
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">{runtime?.modelProviderLabel ?? "Model"} playground</p>
          <p className="font-display text-3xl">Playground</p>
          <p className="mt-2 text-sm text-black/60">
            Compare two models side by side using the same prompt. This works well for cheap local iteration before you
            switch back to paid training later.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <input
            className="rounded-2xl border border-black/10 bg-white px-4 py-3"
            value={baseModel}
            onChange={(event) => setBaseModel(event.target.value)}
            placeholder={runtime?.defaultBaseModel ?? "Base model"}
          />
          <input
            className="rounded-2xl border border-black/10 bg-white px-4 py-3"
            value={fineTunedModel}
            onChange={(event) => setFineTunedModel(event.target.value)}
            placeholder="Second model (optional)"
          />
        </div>

        <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Enter the prompt you want to test..." />

        {error ? <ErrorAlert title="Could not run prompt" description={error} /> : null}

        <Button disabled={loading || !prompt.trim()} onClick={runPrompt}>
          {loading ? "Running..." : "Run comparison"}
        </Button>
      </Card>

      <PlaygroundComparison baseOutput={baseOutput} tunedOutput={tunedOutput} />
    </div>
  );
}
