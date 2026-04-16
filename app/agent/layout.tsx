import { AppShell } from "@/components/app-shell";

export default async function AgentLayout({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
