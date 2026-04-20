import { DashboardClient, type DashboardSummary } from "@/components/dashboard/dashboard-client";
import { getPythonApiBaseUrl } from "@/lib/python-api";

async function loadDashboardSummary() {
  try {
    const response = await fetch(`${getPythonApiBaseUrl()}/dashboard/summary`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    return { data: (await response.json()) as DashboardSummary, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Could not load the fine-tuning studio.",
    };
  }
}

export default async function DashboardPage() {
  const { data, error } = await loadDashboardSummary();
  return <DashboardClient initialData={data} initialError={error} />;
}
