"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div style={{ padding: "40px", fontFamily: "sans-serif", background: "#0f172a", color: "#fff", minHeight: "100vh" }}>
      <h2>Error loading scene</h2>
      <p style={{ color: "#94a3b8" }}>{error?.message || "An unexpected error occurred."}</p>
      <button
        onClick={() => reset()}
        style={{ padding: "8px 16px", background: "#10b981", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer", marginTop: "16px" }}
      >
        Retry
      </button>
    </div>
  );
}
