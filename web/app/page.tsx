"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import AgentPipeline from "./AgentPipeline";
import DashboardNav from "./DashboardNav";
import { fetchVigilSession, localSession, type VigilSession } from "./demoSession";

const BACKEND = process.env.NEXT_PUBLIC_VIGIL_URL || "http://localhost:8000";

const dashboards = [
  {
    index: "01",
    href: "/clinical",
    title: "Clinical Command Center",
    roles: "Charge nurse / triage / attending",
    description: "Ranked queue, multimodal evidence, re-triage decisions, escalation, and documentation.",
    action: "Open clinical",
  },
  {
    index: "02",
    href: "/operations",
    title: "Waiting Room Operations",
    roles: "Front desk / security",
    description: "Seat-level awareness and medical-assist routing with minimum-necessary access.",
    action: "Open operations",
  },
  {
    index: "03",
    href: "/trust",
    title: "Trust & Audit",
    roles: "Compliance",
    description: "Hash-chain verification, access history, redaction, and emergency-access controls.",
    action: "Open trust view",
  },
];

export default function JudgeDemo() {
  const [session, setSession] = useState<VigilSession>(() => localSession("charge_nurse", 0));
  const [capabilities, setCapabilities] = useState<Record<string, boolean>>({
    multi_patient: true,
    role_redaction: true,
    audit_chain: true,
    demo_replay: true,
  });

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2200);

    void Promise.all([
      fetchVigilSession(BACKEND, "charge_nurse"),
      fetch(`${BACKEND}/health`, { cache: "no-store", signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error("Health request unavailable");
        const payload = await response.json();
        return (payload.capabilities ?? {}) as Record<string, boolean>;
      }),
    ])
      .then(([nextSession, nextCapabilities]) => {
        if (!active) return;
        setSession(nextSession);
        setCapabilities(nextCapabilities);
      })
      .catch(() => undefined)
      .finally(() => window.clearTimeout(timeout));

    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, []);

  const activeAlerts = session.queue.filter((patient) => patient.alert).length;

  return (
    <div className="app-shell overview-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <DashboardNav current="overview" />

      <main className="judge-overview">
        <section className="judge-hero surface">
          <div className="judge-hero-copy">
            <span className="eyebrow">Continuous safety between triage and treatment</span>
            <h1>Catch deterioration before the wait becomes the risk.</h1>
            <p>
              Vigil monitors consenting waiting-room patients, corroborates change across signals,
              re-ranks urgency, closes the nurse response loop, and leaves a verifiable clinical record.
            </p>
            <div className="judge-actions">
              <Link className="button-link primary" href="/clinical">Start the judge demo</Link>
              <Link className="button-link" href="/trust">Inspect the audit trail</Link>
            </div>
          </div>

          <div className="judge-system-card inset-surface">
            <div className="system-card-topline">
              <span>Live system state</span>
              <strong className={session.source === "backend" ? "connected" : "preview"}>
                {session.source === "backend" ? "Backend connected" : "Preview mode"}
              </strong>
            </div>
            <div className="overview-metrics">
              <div><strong>{session.queue.length}</strong><span>patients monitored</span></div>
              <div><strong>{activeAlerts}</strong><span>active escalations</span></div>
              <div><strong>{session.audit_verified.blocks}</strong><span>verified audit blocks</span></div>
              <div><strong>6</strong><span>protected roles</span></div>
            </div>
            <div className="overview-priority">
              <span>Highest current priority</span>
              <strong>{session.queue[0]?.name ?? "Identity protected"}</strong>
              <small>
                {session.queue[0]?.current_esi ? `ESI ${session.queue[0].current_esi}` : "Role-filtered"}
                {session.queue[0]?.seat ? ` / Seat ${session.queue[0].seat}` : ""}
              </small>
            </div>
          </div>
        </section>

        <AgentPipeline capabilities={capabilities} source={session.source} />

        <section className="dashboard-directory" aria-labelledby="dashboard-heading">
          <div className="directory-heading">
            <div>
              <span className="eyebrow">Three views, one source of truth</span>
              <h2 id="dashboard-heading">Purpose-built for the people responding</h2>
            </div>
            <p>Every screen reads the same patient registry and audit chain. The backend removes fields a role cannot access.</p>
          </div>
          <div className="dashboard-card-grid">
            {dashboards.map((dashboard) => (
              <Link className="dashboard-link-card surface" href={dashboard.href} key={dashboard.href}>
                <span className="dashboard-card-index">{dashboard.index}</span>
                <div>
                  <h3>{dashboard.title}</h3>
                  <small>{dashboard.roles}</small>
                  <p>{dashboard.description}</p>
                </div>
                <strong>{dashboard.action}<i aria-hidden="true">→</i></strong>
              </Link>
            ))}
          </div>
        </section>

        <section className="judge-flow surface">
          <span className="eyebrow">The sixty-second story</span>
          <div className="judge-flow-grid">
            {[
              ["Observe", "A patient deviates from their own baseline."],
              ["Corroborate", "Independent visual and audio signals increase confidence."],
              ["Re-triage", "Chart risk and deterministic safety floors raise urgency."],
              ["Respond", "The right staff member receives and acknowledges the alert."],
              ["Document", "SOAP, FHIR Provenance, and the audit chain preserve the record."],
            ].map(([title, copy], index) => (
              <article key={title}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{title}</strong>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
