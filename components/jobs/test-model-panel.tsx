"use client";

import { useRef, useState } from "react";
import { Bot, ChevronDown, ChevronUp, Send, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { pythonApiFetch } from "@/lib/python-api";

type Message = { role: "user" | "assistant"; content: string };

export function TestModelPanel({
  fineTunedModel,
  modelProvider,
  baseModel,
  ollamaModelName,
  ollamaRegistered,
}: {
  fineTunedModel: string;
  modelProvider: string;
  baseModel: string;
  ollamaModelName: string | null;
  ollamaRegistered: boolean;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [showSystem, setShowSystem] = useState(false);
  const [ollamaName, setOllamaName] = useState(ollamaModelName ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isLocal = modelProvider === "local";
  const effectiveModel = isLocal ? ollamaName.trim() : fineTunedModel;
  const effectiveProvider = isLocal ? "ollama" : modelProvider;
  const canSend = input.trim().length > 0 && !loading && (!isLocal || ollamaName.trim().length > 0);

  async function sendMessage() {
    if (!canSend) return;
    const userMsg: Message = { role: "user", content: input.trim() };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const result = await pythonApiFetch<{ tunedOutput: string | null; baseOutput: string | null }>(
        "/playground/run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: userMsg.content,
            systemPrompt,
            messages: history,
            singleTurn: true,
            baseModel: effectiveModel,
            baseModelProvider: effectiveProvider,
          }),
        }
      );
      const reply = result.baseOutput ?? "(no response)";
      setMessages([...history, { role: "assistant", content: reply }]);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
      setMessages(history);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="space-y-3">
      {/* Ollama model name input for local jobs */}
      {isLocal && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 space-y-2">
          <p className="text-sm font-medium text-black/80">Ollama model name required</p>
          <p className="text-xs text-black/55 leading-relaxed">
            {ollamaModelName
              ? ollamaRegistered
                ? <>Model <span className="font-mono text-black/70">{ollamaModelName}</span> was pushed to Ollama during training and is pre-filled below.</>
                : <>Suggested Ollama model name <span className="font-mono text-black/70">{ollamaModelName}</span> is pre-filled below. If registration failed, start Ollama and create it manually before testing.</>
              : <>Local adapters run via Ollama. Enter the Ollama model name, or run <span className="font-mono bg-black/6 px-1 py-0.5 rounded">ollama list</span> to find it.</>
            }
          </p>
          <input
            type="text"
            placeholder="e.g. my-fine-tuned-model"
            value={ollamaName}
            onChange={(e) => setOllamaName(e.target.value)}
            className="w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </div>
      )}

      {/* System prompt toggle */}
      <button
        type="button"
        onClick={() => setShowSystem((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-black/45 hover:text-black/70 transition"
      >
        {showSystem ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        System prompt {showSystem ? "— click to hide" : "(optional)"}
      </button>

      {showSystem && (
        <textarea
          rows={2}
          placeholder="e.g. You are a helpful assistant."
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          className="w-full resize-none rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand/40"
        />
      )}

      {/* Chat history */}
      {messages.length > 0 && (
        <div className="max-h-[400px] overflow-y-auto rounded-2xl border border-black/8 bg-white p-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
              <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-white ${
                msg.role === "user" ? "bg-brand" : "bg-black/20"
              }`}>
                {msg.role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
              </div>
              <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-brand/10 text-black/80"
                  : "bg-black/5 text-black/75"
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-black/20 text-white">
                <Bot className="h-3.5 w-3.5" />
              </div>
              <div className="rounded-2xl bg-black/5 px-4 py-2.5">
                <span className="inline-flex gap-1 text-black/40 text-xs">
                  <span className="animate-bounce" style={{ animationDelay: "0ms" }}>●</span>
                  <span className="animate-bounce" style={{ animationDelay: "150ms" }}>●</span>
                  <span className="animate-bounce" style={{ animationDelay: "300ms" }}>●</span>
                </span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      {error && <p className="text-sm text-danger">{error}</p>}

      <p className="text-xs text-black/45">
        Each send is tested independently. Previous turns stay visible here for reference, but only your latest message is sent to the model.
      </p>

      {/* Input row */}
      <div className="flex gap-2">
        <textarea
          rows={2}
          placeholder="Type a message… (Enter to send, Shift+Enter for new line)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          className="flex-1 resize-none rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand/40 disabled:opacity-50"
        />
        <Button
          onClick={sendMessage}
          disabled={!canSend}
          className="self-end px-4"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>

      {messages.length > 0 && (
        <button
          type="button"
          onClick={() => setMessages([])}
          className="text-xs text-black/35 hover:text-black/60 transition"
        >
          Clear conversation
        </button>
      )}
    </div>
  );
}
