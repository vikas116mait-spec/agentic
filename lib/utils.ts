import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(value: Date | string | null | undefined) {
  if (!value) {
    return "--";
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

export function formatNumber(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "--";
  }

  return new Intl.NumberFormat("en-US").format(value);
}

export function formatBytes(bytes: number | null | undefined) {
  if (!bytes && bytes !== 0) {
    return "--";
  }

  const units = ["B", "KB", "MB", "GB"];
  let current = bytes;
  let index = 0;

  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }

  return `${current.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "--";
  }

  const totalSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

export function slugifyFileName(value: string) {
  return value.replace(/[^a-zA-Z0-9.\-_]/g, "-").toLowerCase();
}
