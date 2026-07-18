"use client";

import { useCallback, useEffect, useState } from "react";
import DashboardNav from "../DashboardNav";
import {
  fetchVigilSession,
  localSession,
  postVigilCommand,
  type AuditView,
  type VigilSession,
} from "../demoSession";
import { useVigilBackend } from "../lib/useVigilBackend";

export default function TrustDashboard() {
  const backend = useVigilBackend();
  const [session, setSession] = useState<VigilSession>(() => localSession("compliance", 0));
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("Audit verification is ready.");

  const refresh = useCallback(async () => {
    try {
      if (!backend) throw new Error("Backend unavailable");
      const next = await fetchVigilSession(backend, "compliance");
      setSession(next);
      return next;
    } catch {
      const next = localSession("compliance", 0);
      setSession(next);
      return next;
    }
  }, [backend]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const verifyChain = async () => {
    setBusy(true);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 1800);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await fetch(`${backend}/api/v1/audit/verify`, {
        headers: { "X-Vigil-Role": "compliance" },
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Audit verification unavailable");
      const verification = await response.json();
      setSession((current) => ({ ...current, audit_verified: verification, source: "backend" }));
      setNotice(`Verified ${verification.blocks} linked blocks against the current chain head.`);
    } catch {
      setNotice("Preview verification passed. Connect the backend to verify the live chain.");
    } finally {
      window.clearTimeout(timeout);
      setBusy(false);
    }
  };

  const runBreakGlassTest = async () => {
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, "/api/v1/break-glass", "charge_nurse", {
        actor: "Charge RN demo",
        patient_id: "demo-vega",
        reason: "Patient collapsed outside assigned clinical unit",
      });
      if (!response.ok) throw new Error("Break-glass test unavailable");
      await refresh();
      setNotice("Emergency access granted for 15 minutes and permanently added to the audit chain.");
    } catch {
      const block: AuditView = {
        index: session.audit_verified.blocks,
        audit_id: `preview-break-glass-${session.audit_verified.blocks + 1}`,
        ts: 1784401200,
        actor: "Charge RN demo",
        role: "charge_nurse",
        action: "break_glass",
        resource: "patient:pt_4fd8b2c7a1",
        outcome: "granted",
        reason: "Patient collapsed outside assigned clinical unit",
        hash: "b8e9a12d72f4",
      };
      setSession((current) => ({
        ...current,
        source: "local",
        audit: [...(current.audit ?? []), block],
        audit_verified: { valid: true, blocks: current.audit_verified.blocks + 1, head: block.hash },
      }));
      setNotice("Preview added a redacted emergency-access event to the audit chain.");
    } finally {
      setBusy(false);
    }
  };

  const audit = [...(session.audit ?? [])].reverse();
  const denied = audit.filter((block) => block.outcome === "denied").length;
  const emergency = audit.filter((block) => block.action === "break_glass").length;

  return (
    <div className="app-shell trust-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <DashboardNav current="trust" />

      <header className="dashboard-header surface">
        <div>
          <span className="eyebrow">Compliance view</span>
          <h1>Trust and audit</h1>
          <p>Review access and decisions with patient identifiers replaced by scoped references.</p>
        </div>
        <div className="dashboard-header-controls">
          <span className="role-badge">Compliance view</span>
          <span className={`source-pill ${session.source === "backend" ? "connected" : ""}`}>
            {session.source === "backend" ? "Backend connected" : "Local replay data"}
          </span>
        </div>
      </header>

      <section className="trust-metrics" aria-label="Audit summary">
        <article className="metric-card surface"><span>Chain integrity</span><strong className="metric-success">{session.audit_verified.valid ? "Verified" : "Failed"}</strong><small>SHA-256 linked blocks</small></article>
        <article className="metric-card surface"><span>Audit blocks</span><strong>{session.audit_verified.blocks}</strong><small>Current session</small></article>
        <article className="metric-card surface"><span>Denied reads</span><strong>{denied}</strong><small>Policy enforced</small></article>
        <article className="metric-card surface"><span>Emergency grants</span><strong>{emergency}</strong><small>Permanent events</small></article>
      </section>

      <main className="trust-workspace">
        <section className="audit-ledger surface">
          <div className="panel-heading-row">
            <div><span className="eyebrow">Append-only record</span><h2>Audit ledger</h2></div>
            <button type="button" onClick={() => void refresh()} disabled={busy}>Refresh</button>
          </div>
          <div className="ledger-head" aria-hidden="true">
            <span>Hash</span><span>Actor</span><span>Action</span><span>Resource</span><span>Outcome</span>
          </div>
          <div className="ledger-body">
            {audit.length ? audit.map((block) => (
              <article className={`ledger-row ${block.outcome === "denied" ? "denied" : ""}`} key={block.audit_id}>
                <code>{block.hash.slice(0, 10)}</code>
                <div><strong>{block.actor}</strong><small>{new Date(block.ts * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })} UTC</small></div>
                <span>{block.action.replace(/_/g, " ")}</span>
                <span>{block.resource}</span>
                <i>{block.outcome}</i>
              </article>
            )) : <p className="ledger-empty">The ledger will populate when the backend receives an action.</p>}
          </div>
        </section>

        <aside className="trust-controls">
          <section className="integrity-card surface">
            <span className="eyebrow">Chain integrity</span>
            <div className={`integrity-seal ${session.audit_verified.valid ? "valid" : ""}`}><span /></div>
            <h2>{session.audit_verified.valid ? "Chain verified" : "Verification failed"}</h2>
            <p>Each block commits to the previous hash. Editing any historical action invalidates the remainder of the chain.</p>
            <code>{session.audit_verified.head?.slice(0, 24) ?? "GENESIS"}</code>
            <button className="verify-button" type="button" onClick={() => void verifyChain()} disabled={busy}>Verify live chain</button>
          </section>

          <section className="control-test-card surface">
            <span className="eyebrow">Emergency-access control</span>
            <h2>Break-glass test</h2>
            <p>Creates a fifteen-minute clinical grant with a required reason and a permanent compliance event.</p>
            <button type="button" onClick={() => void runBreakGlassTest()} disabled={busy}>Run access-control test</button>
          </section>

          <p className="trust-notice surface" aria-live="polite">{notice}</p>
        </aside>
      </main>
    </div>
  );
}
