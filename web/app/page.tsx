"use client";

import { useMemo, useState } from "react";
import DashboardNav from "./DashboardNav";
import { postVigilCommand, type QueuePatient } from "./demoSession";
import { useRoleSession } from "./useRoleSession";

function SectionTitle({ index, title, detail }: { index: string; title: string; detail?: string }) {
  return <div className="section-title"><span>{index}</span><div><h2>{title}</h2>{detail ? <p>{detail}</p> : null}</div></div>;
}

function patientKey(patient: QueuePatient) {
  return patient.patient_id ?? patient.patient_ref ?? patient.seat ?? "patient";
}

export default function CommandCenterPage() {
  const { backend, session, loading, refresh } = useRoleSession("charge_nurse", 1800);
  const [selectedKey, setSelectedKey] = useState("demo-vega");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("System is continuously monitoring all consented patients.");
  const selected = useMemo(
    () => session.queue.find((item) => patientKey(item) === selectedKey) ?? session.queue[0],
    [selectedKey, session.queue],
  );
  const alert = selected?.alert;

  const command = async (path: string, body: Record<string, unknown>, success: string) => {
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, path, "charge_nurse", body);
      if (!response.ok) throw new Error(await response.text());
      setNotice(success);
      await refresh();
    } catch {
      setNotice("Live command unavailable. The local preview remains active.");
    } finally {
      setBusy(false);
    }
  };

  const activeAlerts = session.queue.filter((item) => item.alert && item.alert.state !== "resolved");
  const audit = [...(session.audit ?? [])].slice(-7).reverse();

  return <div className="app-shell">
    <DashboardNav current="command" />
    <main className="command-grid">
      <section className="panel command-observation">
        <SectionTitle index="01" title="Live observation" detail="Derived signals only" />
        <div className="camera-stage">
          <div className="camera-label"><span>CAM 01</span><span>WAITING ROOM</span></div>
          <div className="camera-gridmark"><span /><strong>{selected?.seat ?? "A3"}</strong><p>Patient track active</p><small>No raw video retained</small></div>
        </div>
        <div className="patient-context">
          <span className="eyebrow">Patient context</span>
          <h3>{selected?.name ?? "Waiting for patient"}</h3>
          <p>{selected?.visit ?? "Select a patient from the queue."}</p>
          <div className="data-grid cols-3">
            <div><small>Seat</small><strong>{selected?.seat ?? "No data"}</strong></div>
            <div><small>Current ESI</small><strong>{selected?.current_esi ?? "No data"}</strong></div>
            <div><small>Wait</small><strong>{selected?.wait_minutes ?? 0} min</strong></div>
          </div>
          <div className="queue-strip">
            {session.queue.map((item) => <button type="button" className={patientKey(item) === patientKey(selected ?? {}) ? "active" : ""} onClick={() => setSelectedKey(patientKey(item))} key={patientKey(item)}><span>{item.seat}</span><strong>{item.name}</strong><small>ESI {item.current_esi} / {item.status}</small></button>)}
          </div>
        </div>
      </section>

      <section className="panel command-reasoning">
        <SectionTitle index="02" title="Clinical reasoning" detail="Chart-grounded review" />
        <div className="state-card"><span className="state-icon">II</span><div><small>Current state</small><strong>{selected?.latest_signal ?? "Continuous monitoring"}</strong></div></div>
        <div className="reasoning-trace">
          <article><i /><div><small>Observation</small><p>{selected?.latest_signal ?? "No event requires review."}</p></div></article>
          <article><i /><div><small>Chart context</small><p>{selected?.risk_factors?.length ? selected.risk_factors.join(" / ") : "No material chart risks added."}</p></div></article>
          <article><i /><div><small>Decision</small><p>{alert ? `${alert.title}. ESI ${alert.prior_esi} to ESI ${alert.current_esi}.` : "Continue scheduled reassessment."}</p></div></article>
        </div>
        <div className="command-controls">
          <button disabled={busy} onClick={() => void command("/api/v1/demo/advance", {}, "Advanced one deterministic sensor event.")}>Advance signal</button>
          <button disabled={busy} onClick={() => void command("/api/v1/demo/reset", {}, "Demo session reset.")}>Reset session</button>
        </div>
      </section>

      <section className="panel command-response">
        <SectionTitle index="03" title="Response" detail="Escalation and record" />
        <div className={`response-state ${alert ? "alert" : ""}`}><i /><div><small>Escalation state</small><h3>{alert?.state.replace(/_/g, " ") ?? "Monitoring"}</h3><p>{alert?.title ?? "No active escalation"}</p></div></div>
        <div className="data-grid cols-2">
          <div><small>Reasoning</small><strong>{alert ? "Complete" : "Standby"}</strong></div>
          <div><small>Nurse call</small><strong>{alert?.state === "page_pending" ? "Pending" : "Standby"}</strong></div>
          <div><small>Patient check-in</small><strong>{selected?.status === "checkin" ? "Active" : "Standby"}</strong></div>
          <div><small>Camera</small><strong>Derived only</strong></div>
        </div>
        <div className="response-actions">
          {alert?.state === "page_pending" ? <button disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/acknowledge`, { actor: "Charge RN" }, "Alert acknowledged by Charge RN.")}>Acknowledge alert</button> : null}
          {alert && ["acknowledged", "checkin"].includes(alert.state) ? <>
            <button disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/feedback`, { actor: "Charge RN", outcome: "confirmed" }, "Incident confirmed and closed.")}>Confirm incident</button>
            <button disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/feedback`, { actor: "Charge RN", outcome: "false_alarm" }, "Event marked as false alarm.")}>False alarm</button>
          </> : null}
        </div>
        <div className="activity-block"><div><span className="eyebrow">Activity</span><small>{loading ? "Syncing" : session.source === "backend" ? "Live" : "Preview"}</small></div><p>{notice}</p></div>
        <div className="activity-block grow"><div><span className="eyebrow">Open alerts</span><small>{activeAlerts.length} active</small></div>{activeAlerts.length ? activeAlerts.map((item) => <button className="alert-row" onClick={() => setSelectedKey(patientKey(item))} key={patientKey(item)}><span>{item.seat}</span><strong>{item.alert?.title}</strong><small>{item.alert?.state.replace(/_/g, " ")}</small></button>) : <p>No escalations in this session.</p>}</div>
      </section>
    </main>
    <section className="panel system-events">
      <SectionTitle index="04" title="System events" detail="Sensor and agent activity" />
      <div className="event-table"><div className="event-head"><span>Time</span><span>Type</span><span>Source</span><span>Event</span><span>Patient</span></div>{audit.length ? audit.map((block) => <div className="event-row" key={block.audit_id}><span>{new Date(block.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span><span>{block.action.includes("alert") ? "Response" : "System"}</span><span>{block.role}</span><span>{block.action.replace(/_/g, " ")} / {block.outcome}</span><span>{block.resource.replace("patient:", "")}</span></div>) : <div className="empty-row">No system events recorded.</div>}</div>
    </section>
  </div>;
}
