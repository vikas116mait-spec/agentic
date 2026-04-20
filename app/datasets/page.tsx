import { DatasetsPageClient, type DatasetListItem } from "@/components/dataset/datasets-page-client";
import { getPythonApiBaseUrl } from "@/lib/python-api";

async function loadDatasets() {
  try {
    const response = await fetch(`${getPythonApiBaseUrl()}/datasets`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    return { data: (await response.json()) as DatasetListItem[], error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Could not load datasets.",
    };
  }
}

export default async function DatasetsPage() {
  const { data, error } = await loadDatasets();
  return <DatasetsPageClient initialDatasets={data} initialError={error} />;
}
