import { CreateJobForm } from "@/components/jobs/create-job-form";
import { Card } from "@/components/ui/card";

export default async function NewJobPage({
  searchParams
}: {
  searchParams: Promise<{ datasetId?: string }>;
}) {
  const params = await searchParams;

  return (
    <Card className="space-y-6">
      <div>
        <p className="font-display text-3xl">Create fine-tuning job</p>
        <p className="mt-2 text-sm text-black/60">
          Pick a validated dataset and base model to launch a supervised fine-tune when the OpenAI provider is enabled.
        </p>
      </div>

      <CreateJobForm initialDatasetId={params.datasetId ?? ""} />
    </Card>
  );
}
