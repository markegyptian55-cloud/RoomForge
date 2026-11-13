"use client";

import React, { useState } from "react";

export interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthSuccess: (token: string, email: string) => void;
}

export default function AuthModal({ isOpen, onClose, onAuthSuccess }: AuthModalProps) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    if (!email || !password) {
      setError("Please fill in both email and password.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }

    setBusy(true);
    const endpoint = isRegister ? "/api/auth/register" : "/api/auth/login";
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Authentication failed (${res.status})`);
      }
      setSuccess(isRegister ? "Account created successfully!" : "Logged in successfully!");
      if (typeof window !== "undefined") {
        localStorage.setItem("roomforge_token", data.access_token);
        localStorage.setItem("roomforge_user_email", email.trim().toLowerCase());
      }
      setTimeout(() => {
        onAuthSuccess(data.access_token, email.trim().toLowerCase());
        onClose();
      }, 500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="User Authentication"
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(8px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
        padding: "16px",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "400px",
          background: "#0f172a",
          border: "1px solid rgba(45, 212, 191, 0.3)",
          borderRadius: "16px",
          padding: "24px",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.8)",
          color: "#f8fafc",
          display: "grid",
          gap: "16px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, margin: 0, color: "#2dd4bf" }}>
            {isRegister ? "Create RoomForge Account" : "Sign In to RoomForge"}
          </h2>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "#94a3b8",
              cursor: "pointer",
              fontSize: "18px",
              padding: "4px",
            }}
          >
            ✕
          </button>
        </div>

        {error ? (
          <div role="alert" style={{ background: "rgba(239, 68, 68, 0.2)", border: "1px solid #ef4444", borderRadius: "8px", padding: "10px", color: "#fca5a5", fontSize: "12px" }}>
            {error}
          </div>
        ) : null}

        {success ? (
          <div role="status" style={{ background: "rgba(34, 197, 94, 0.2)", border: "1px solid #22c55e", borderRadius: "8px", padding: "10px", color: "#86efac", fontSize: "12px" }}>
            {success}
          </div>
        ) : null}

        <form onSubmit={handleSubmit} style={{ display: "grid", gap: "12px" }}>
          <label style={{ display: "grid", gap: "4px", fontSize: "12px" }}>
            <span>Email Address</span>
            <input
              type="email"
              required
              value={email}
              placeholder="user@example.com"
              onChange={(e) => setEmail(e.target.value)}
              style={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                padding: "10px 12px",
                color: "#f8fafc",
                fontSize: "14px",
              }}
            />
          </label>

          <label style={{ display: "grid", gap: "4px", fontSize: "12px" }}>
            <span>Password (min 8 chars)</span>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              placeholder="••••••••"
              onChange={(e) => setPassword(e.target.value)}
              style={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                padding: "10px 12px",
                color: "#f8fafc",
                fontSize: "14px",
              }}
            />
          </label>

          <button
            type="submit"
            disabled={busy}
            style={{
              background: "#2dd4bf",
              color: "#0f172a",
              border: "none",
              borderRadius: "8px",
              padding: "11px",
              fontWeight: 700,
              fontSize: "14px",
              cursor: busy ? "not-allowed" : "pointer",
              marginTop: "4px",
            }}
          >
            {busy ? "Authenticating..." : isRegister ? "Sign Up" : "Sign In"}
          </button>
        </form>

        <div style={{ textAlign: "center", fontSize: "12px", color: "#94a3b8" }}>
          {isRegister ? "Already have an account? " : "Don't have an account yet? "}
          <button
            type="button"
            onClick={() => {
              setIsRegister(!isRegister);
              setError("");
              setSuccess("");
            }}
            style={{
              background: "transparent",
              border: "none",
              color: "#38bdf8",
              cursor: "pointer",
              fontWeight: 600,
              textDecoration: "underline",
              padding: 0,
            }}
          >
            {isRegister ? "Sign In" : "Sign Up"}
          </button>
        </div>
      </div>
    </div>
  );
}
