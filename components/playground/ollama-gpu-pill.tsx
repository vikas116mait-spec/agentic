"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { pythonApiFetch } from "@/lib/python-api";

type GpuEntry = {
  index: number;
  name: string;
  memoryTotalMb: number;
  memoryUsedMb: number;
  memoryFreeMb: number;
  utilizationPct: number;
  temperatureC: number;
};

type GpuStatus = {
  gpus: GpuEntry[];
  pinnedGpu: number | null;
  pinnedGpuFreeMb: number | null;
  recommendedGpu: number | null;
  recommendedGpuFreeMb: number | null;
  autoManageEnabled: boolean;
  dropinWritable: boolean;
  sudoAvailable: boolean;
  nvidiaSmiAvailable: boolean;
  minFreeGpuMb: number;
  serviceName: string;
};

type RebalanceResponse = {
  result: { changed: boolean; gpuIndex: number; freeMb: number; previousGpu?: number | null; reason?: string };
  status: GpuStatus;
};

function formatGb(mb: number | null | undefined): string {
  if (mb === null || mb === undefined) return "-";
  return `${(mb / 1024).toFixed(1)} GB`;
}

export function OllamaGpuPill() {
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [rebalancing, setRebalancing] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const payload = await pythonApiFetch<GpuStatus>("/settings/gpu/inventory");
      setStatus(payload);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load GPU status.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (loading) return null;
  if (!status) return null;
  if (!status.autoManageEnabled) return null;

  const setupBlocked = !status.dropinWritable || !status.sudoAvailable || !status.nvidiaSmiAvailable;

  async function rebalance() {
    setRebalancing(true);
    setMessage(null);
    setError(null);
    try {
      const payload = await pythonApiFetch<RebalanceResponse>("/settings/gpu/rebalance-ollama", {
        method: "POST",
      });
      setStatus(payload.status);
      if (payload.result.changed) {
        setMessage(`Moved Ollama to GPU ${payload.result.gpuIndex} (~${formatGb(payload.result.freeMb)} free).`);
      } else {
        setMessage(`Ollama already on GPU ${payload.result.gpuIndex} (~${formatGb(payload.result.freeMb)} free).`);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Rebalance failed.");
    } finally {
      setRebalancing(false);
    }
  }

  const pinnedLabel =
    status.pinnedGpu === null
      ? "Ollama GPU: auto (default GPU 0)"
      : `Ollama on GPU ${status.pinnedGpu} \u00b7 ${formatGb(status.pinnedGpuFreeMb)} free`;

  const recommendedLabel =
    status.recommendedGpu !== null && status.recommendedGpu !== status.pinnedGpu
      ? `Freest: GPU ${status.recommendedGpu} (~${formatGb(status.recommendedGpuFreeMb)})`
      : null;

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-black/10 bg-white px-4 py-2 text-xs">
      <span className="inline-flex items-center gap-2">
        <span
          className="h-2 w-2 rounded-full"
          style={{
            backgroundColor:
              status.pinnedGpuFreeMb !== null && status.pinnedGpuFreeMb >= status.minFreeGpuMb
                ? "#16a34a"
                : "#dc2626",
          }}
        />
        <span className="font-medium text-black/80">{pinnedLabel}</span>
      </span>
      {recommendedLabel ? <span className="text-black/55">{recommendedLabel}</span> : null}
      <Button size="sm" variant="ghost" onClick={rebalance} disabled={rebalancing || setupBlocked}>
        {rebalancing ? "Rebalancing..." : "Move to freest GPU"}
      </Button>
      {setupBlocked ? (
        <span className="text-black/55">
          {!status.nvidiaSmiAvailable
            ? "nvidia-smi not found"
            : !status.dropinWritable
            ? "Drop-in not writable"
            : "Passwordless sudo missing"}
          {" "}- see README.
        </span>
      ) : null}
      {message ? <span className="text-black/65">{message}</span> : null}
      {error ? <span className="text-red-600">{error}</span> : null}
    </div>
  );
}
