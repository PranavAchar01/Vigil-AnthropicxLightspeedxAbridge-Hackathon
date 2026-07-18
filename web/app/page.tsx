"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import AgentPipeline from "./AgentPipeline";
import BackendFeed from "./BackendFeed";
import DashboardNav from "./DashboardNav";
import { fetchVigilSession, localSession, type VigilSession } from "./demoSession";
import { useVigilBackend } from "./lib/useVigilBackend";

const dashboards = [
  {
    href: "/clinical",
    label: "Clinical",
    title: "Re-triage and response",
    copy: "Review the ranked queue, inspect chart-grounded reasoning, and close an alert.",
    meta: "Charge nurse and clinical staff",
  },
  {
    href: "/operations",
    label: "Operations",
    title: "Seat-level coordination",
    copy: "Route a medical-assist request without exposing diagnoses, ESI, or camera data.",
    meta: "Front desk and security",
  },
  {
    href: "/trust",
    label: "Trust",
    title: "Access and audit record",
    copy: "Verify the linked audit chain and test time-limited emergency access.",
    meta: "Compliance",
  },
];

export default function OverviewPage() {
  const backend = useVigilBackend();
  const [session, setSession] = useState<VigilSession>(() => localSession("charge_nurse", 2));
  const [capabilities, setCapabilities] = useState<Record<string, boolean>>({
    multi_patient: true,
    audit_chain: true,
    demo_replay: true,
    role_redaction: true,
  });

  useEffect(() => {
    if (!backend) return;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2200);
    void Promise.all([
      fetchVigilSession(backend, "charge_nurse"),
      fetch(`${backend}/health`, { cache: "no-store", signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error("Health request unavailable");
        const payload = await response.json();
        return (payload.capabilities ?? {}) as Record<string, boolean>;
      }),
    ])
      .then(([nextSession, nextCapabilities]) => {
        setSession(nextSession);
        setCapabilities(nextCapabilities);
      })
      .catch(() => undefined)
      .finally(() => window.clearTimeout(timeout));
    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [backend]);

  const activeAlerts = session.queue.filter((patient) => patient.alert).length;
  const topPatient = session.queue[0];

  return (
    <div className="app-shell overview-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <DashboardNav current="overview" />

      <main className="judge-overview">
        <section className="judge-hero surface">
          <div className="judge-hero-copy">
            <span className="eyebrow">Continuous re-triage</span>
            <h1>A waiting room that can notice when a patient changes.</h1>
            <p>Vigil watches consented patients after intake. It compares new signals with the chart, updates the queue, contacts staff, and records the response.</p>
            <div className="judge-actions">
              <Link className="button-link primary" href="/clinical">Run the clinical demo</Link>
              <Link className="button-link" href="/operations">Open operations</Link>
            </div>
          </div>

          <div className="judge-system-card inset-surface">
            <div className="system-card-topline">
              <span>Current session</span>
              <strong className={session.source === "backend" ? "connected" : "preview"}>{session.source === "backend" ? "Backend connected" : "Replay data"}</strong>
            </div>
            <div className="overview-priority">
              <span>First in queue</span>
              <strong>{topPatient?.name ?? "Identity protected"}</strong>
              <small>{topPatient?.current_esi ? `ESI ${topPatient.current_esi}` : "Clinical fields restricted"}{topPatient?.seat ? ` / Seat ${topPatient.seat}` : ""}</small>
            </div>
            <div className="overview-metrics">
              <div><strong>{session.queue.length}</strong><span>patients</span></div>
              <div><strong>{activeAlerts}</strong><span>open alerts</span></div>
              <div><strong>{session.audit_verified.blocks}</strong><span>audit blocks</span></div>
            </div>
          </div>
        </section>

        <section className="dashboard-directory" aria-labelledby="dashboard-heading">
          <div className="directory-heading">
            <div><span className="eyebrow">Demo views</span><h2 id="dashboard-heading">One backend, three staff views</h2></div>
            <p>The API filters each response by role. The dashboards do not hide restricted fields after they arrive.</p>
          </div>
          <div className="dashboard-card-grid">
            {dashboards.map((dashboard, index) => (
              <Link className="dashboard-link-card surface" href={dashboard.href} key={dashboard.href}>
                <span className="dashboard-card-index">0{index + 1}</span>
                <div><small>{dashboard.label}</small><h3>{dashboard.title}</h3><p>{dashboard.copy}</p></div>
                <strong>{dashboard.meta}<i aria-hidden="true">Open</i></strong>
              </Link>
            ))}
          </div>
        </section>

        <AgentPipeline capabilities={capabilities} source={session.source} />

        <section className="judge-flow surface">
          <div className="directory-heading"><div><span className="eyebrow">Demo sequence</span><h2>What the judges can try</h2></div><p>Use the replay when a camera or external service is unavailable. The same command API receives each action.</p></div>
          <div className="judge-flow-grid">
            {["Advance a patient signal", "Review the ESI change", "Acknowledge the alert", "Verify the audit block"].map((item, index) => <article key={item}><span>0{index + 1}</span><strong>{item}</strong></article>)}
          </div>
        </section>

        <BackendFeed audit={session.audit} />
      </main>
    </div>
  );
}
