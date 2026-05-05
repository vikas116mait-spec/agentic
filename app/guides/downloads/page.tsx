import Link from "next/link";
import { Archive, Database, Download, FileCog, TerminalSquare } from "lucide-react";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const downloadTypes = [
  {
    id: "dataset-file",
    icon: Database,
    eyebrow: "Dataset file",
    title: "Use it for another training run",
    copy:
      "A dataset download is the original JSONL file. Re-upload it, edit it, validate it again, or hand it to another fine-tuning pipeline."
  },
  {
    id: "adapter-bundle",
    icon: Archive,
    eyebrow: "Adapter bundle",
    title: "Use it with PEFT on top of the base model",
    copy:
      "The adapter zip is a LoRA bundle. Keep the same base model from the metadata, then load the adapter on top of it for inference or recovery."
  },
  {
    id: "gguf-file",
    icon: FileCog,
    eyebrow: "GGUF model",
    title: "Use it with Ollama or llama.cpp",
    copy:
      "The GGUF download is the easiest local artifact when you want a single model file that is ready for Ollama-style serving."
  }
];

export default function DownloadGuidePage() {
  return (
    <div className="space-y-6">
      <Breadcrumb crumbs={[{ label: "Dashboard", href: "/dashboard" }, { label: "Guides" }, { label: "Using downloads" }]} />

      <Card className="overflow-hidden p-0">
        <div className="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(255,247,229,0.9))] p-7 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-brand/15 bg-brand/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand">
              <Download className="h-3.5 w-3.5" />
              After Download
            </div>

            <h1 className="mt-5 max-w-3xl font-display text-4xl leading-tight sm:text-5xl">
              Downloaded the file. Here is what to do next.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-black/62 sm:text-base">
              This app can give you three different download types: the training dataset, the LoRA adapter bundle, and the GGUF model.
              Each one is useful, but each one is used differently.
            </p>

            <div className="mt-6 flex flex-wrap gap-2">
              <Link href="#dataset-file">
                <Button size="sm" variant="secondary">Dataset</Button>
              </Link>
              <Link href="#adapter-bundle">
                <Button size="sm" variant="ghost">Adapter</Button>
              </Link>
              <Link href="#gguf-file">
                <Button size="sm" variant="ghost">GGUF</Button>
              </Link>
            </div>
          </div>

          <div className="bg-[linear-gradient(180deg,rgba(12,29,28,0.98),rgba(25,64,57,0.94))] p-7 text-white sm:p-8">
            <p className="text-xs uppercase tracking-[0.22em] text-white/55">Quick pick</p>
            <div className="mt-4 space-y-3">
              <div className="rounded-[1.3rem] border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                Want to retrain or edit examples: download the dataset.
              </div>
              <div className="rounded-[1.3rem] border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                Want a portable LoRA artifact for Python or PEFT: download the adapter bundle.
              </div>
              <div className="rounded-[1.3rem] border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                Want the easiest local runtime for Ollama: download the GGUF.
              </div>
            </div>
            <p className="mt-5 text-sm leading-7 text-white/68">
              If you are unsure, start with GGUF for local usage and keep the adapter bundle for backup.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        {downloadTypes.map((item) => {
          const Icon = item.icon;

          return (
            <Card key={item.id} className="space-y-4 bg-white/88">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-black/42">{item.eyebrow}</p>
                <p className="mt-2 font-display text-2xl">{item.title}</p>
              </div>
              <p className="text-sm leading-7 text-black/58">{item.copy}</p>
            </Card>
          );
        })}
      </div>

      <Card id="dataset-file" className="space-y-5 bg-white/90 scroll-mt-24">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Dataset file</p>
            <p className="mt-1 font-display text-3xl">How to use a downloaded dataset</p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 1</p>
            <p className="mt-2 text-sm text-black/75">Open the `.jsonl` file and confirm it is still one JSON object per line.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 2</p>
            <p className="mt-2 text-sm text-black/75">Edit examples if needed, but keep the same schema shape for every row.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 3</p>
            <p className="mt-2 text-sm text-black/75">Upload it back into this app and validate it again before training.</p>
          </div>
        </div>

        <div className="rounded-2xl bg-white p-5">
          <p className="font-medium text-black/85">Common uses</p>
          <p className="mt-2 text-sm leading-7 text-black/70">
            Use the dataset download when you want to version your examples, share them with another teammate, fix bad rows,
            or use the same training file in another fine-tuning tool. This is the raw training data, not the trained model.
          </p>
        </div>

        <div className="rounded-2xl bg-black/95 p-5 text-sm text-white/88">
          <p className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-white/55">
            <TerminalSquare className="h-3.5 w-3.5" />
            Example rows
          </p>
          <pre className="overflow-x-auto whitespace-pre-wrap leading-7">{`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Hello! How can I help?"}]}
{"instruction":"Summarize this note","input":"The server restarted successfully.","output":"The server restarted successfully."}`}</pre>
        </div>

        <div className="flex flex-wrap gap-3">
          <Link href="/datasets/new">
            <Button>Upload a dataset</Button>
          </Link>
          <Link href="/jobs/new">
            <Button variant="secondary">Start a training run</Button>
          </Link>
        </div>
      </Card>

      <Card id="adapter-bundle" className="space-y-5 bg-white/90 scroll-mt-24">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
            <Archive className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Adapter bundle</p>
            <p className="mt-1 font-display text-3xl">How to use a downloaded adapter zip</p>
          </div>
        </div>

        <div className="rounded-2xl bg-white p-5">
          <p className="font-medium text-black/85">What is inside</p>
          <p className="mt-2 text-sm leading-7 text-black/70">
            The adapter download is a zip bundle. After unzip, you should see `agentic-metadata.json`, an `adapter/` folder,
            and usually `metrics.json` and `status.json`. The metadata file tells you which base model the adapter belongs to.
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 1</p>
            <p className="mt-2 text-sm text-black/75">Unzip the bundle and read `agentic-metadata.json`.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 2</p>
            <p className="mt-2 text-sm text-black/75">Load the same base model named in that metadata.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 3</p>
            <p className="mt-2 text-sm text-black/75">Apply the adapter on top of the base model with PEFT.</p>
          </div>
        </div>

        <div className="rounded-2xl bg-black/95 p-5 text-sm text-white/88">
          <p className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-white/55">
            <TerminalSquare className="h-3.5 w-3.5" />
            Python example
          </p>
          <pre className="overflow-x-auto whitespace-pre-wrap leading-7">{`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "the-base-model-from-agentic-metadata.json"
adapter_dir = "./adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_dir)`}</pre>
        </div>

        <div className="rounded-2xl bg-white p-5">
          <p className="font-medium text-black/85">Important</p>
          <p className="mt-2 text-sm leading-7 text-black/70">
            An adapter bundle is not a standalone Ollama model. If your goal is local serving with Ollama, download the GGUF instead.
            If you only have the adapter and still want an Ollama model, use the original job workspace and run the recovery/export flow there.
          </p>
        </div>
      </Card>

      <Card id="gguf-file" className="space-y-5 bg-white/90 scroll-mt-24">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand/10 text-brand">
            <FileCog className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">GGUF model</p>
            <p className="mt-1 font-display text-3xl">How to use a downloaded GGUF</p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 1</p>
            <p className="mt-2 text-sm text-black/75">Save the `.gguf` file somewhere stable on disk.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 2</p>
            <p className="mt-2 text-sm text-black/75">Create a simple `Modelfile` that points to that absolute file path.</p>
          </div>
          <div className="rounded-2xl bg-muted/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Step 3</p>
            <p className="mt-2 text-sm text-black/75">Register it with Ollama, then run it like any other local model.</p>
          </div>
        </div>

        <div className="rounded-2xl bg-black/95 p-5 text-sm text-white/88">
          <p className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-white/55">
            <TerminalSquare className="h-3.5 w-3.5" />
            Ollama example
          </p>
          <pre className="overflow-x-auto whitespace-pre-wrap leading-7">{`FROM /absolute/path/to/your-model.gguf
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|eot_id|>"`}</pre>
          <pre className="mt-4 overflow-x-auto whitespace-pre-wrap leading-7">{`ollama create my-finetune -f Modelfile
ollama run my-finetune`}</pre>
        </div>

        <div className="rounded-2xl bg-white p-5">
          <p className="font-medium text-black/85">Why GGUF is the easiest local download</p>
          <p className="mt-2 text-sm leading-7 text-black/70">
            GGUF is already a merged, portable model artifact for local runtimes. You do not need to manually attach a LoRA adapter first.
            That is why GGUF is the best choice when your next step is “I just want to run this model locally.”
          </p>
        </div>
      </Card>
    </div>
  );
}
