"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";

function subscribe(callback: () => void): () => void {
  window.addEventListener("storage", callback);
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", callback);
  return () => {
    window.removeEventListener("storage", callback);
    media.removeEventListener("change", callback);
  };
}

function getSnapshot(): boolean {
  if (typeof window === "undefined") return false;
  const stored = localStorage.getItem("room3d_theme");
  if (stored !== null) return stored === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function getServerSnapshot(): boolean {
  return false;
}

export function ThemeToggle() {
  const isDark = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDark]);

  const toggle = () => {
    const next = !isDark;
    localStorage.setItem("room3d_theme", next ? "dark" : "light");
    if (next) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    window.dispatchEvent(new Event("storage"));
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle color theme"
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: "8px",
        padding: "6px 12px",
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        color: "var(--ink)",
        fontSize: "12px",
        fontWeight: 600,
      }}
    >
      {isDark ? <Sun size={14} /> : <Moon size={14} />}
      <span>{isDark ? "Light Mode" : "Dark Mode"}</span>
    </button>
  );
}
