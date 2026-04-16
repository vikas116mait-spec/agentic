import { Card } from "@/components/ui/card";

export default function SettingsPage() {
  return (
    <Card>
      <p className="font-display text-3xl">Settings</p>
      <p className="mt-3 text-sm text-black/60">
        The interface talks to the local Python API on port `8001`, and the agentic flow runs through Temporal. For
        near-free local use, set `LLM_PROVIDER=ollama`, make sure Ollama is running on `http://127.0.0.1:11434`, and pull
        a model such as `qwen3:8b`. When you are ready to switch back later, set `LLM_PROVIDER=openai` and add
        `OPENAI_API_KEY` in `.env`.
      </p>
    </Card>
  );
}
