import { JobDetailClient } from "@/components/jobs/job-detail-client";

export default async function JobDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <JobDetailClient jobId={id} />;
}
