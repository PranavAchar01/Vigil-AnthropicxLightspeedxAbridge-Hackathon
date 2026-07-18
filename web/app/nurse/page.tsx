"use client";

import { useMemo, useState } from "react";
import DashboardNav from "../DashboardNav";
import { postVigilCommand, type QueuePatient } from "../demoSession";
import { useRoleSession } from "../useRoleSession";

const keyFor = (patient: QueuePatient) => patient.patient_id ?? patient.seat ?? "patient";

export default function NursePage() {
  const { backend, session, refresh } = useRoleSession("charge_nurse", 1800);
  const [selectedKey, setSelectedKey] = useState("demo-vega");
  const [note, setNote] = useState("");
  const [notice, setNotice] = useState("The queue is ranked by acuity, observed change, chart risk, and wait time.");
  const [busy, setBusy] = useState(false);
  const selected = useMemo(() => session.queue.find((item) => keyFor(item) === selectedKey) ?? session.queue[0], [selectedKey, session.queue]);

  const command = async (path: string, body: Record<string, unknown>, success: string) => {
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, path, "charge_nurse", body);
      if (!response.ok) throw new Error(await response.text());
      setNotice(success);
      await refresh();
    } catch {
      setNotice("Live command unavailable. The clinical preview remains visible.");
    } finally { setBusy(false); }
  };

  const alert = selected?.alert;
  const vitals = Object.entries(selected?.chart?.vitals ?? {});

  return <div className="app-shell">
    <DashboardNav current="nurse" />
    <header className="crm-header panel"><div><span className="eyebrow">Charge nurse CRM</span><h1>Clinical response queue</h1><p>Review patient change, acknowledge escalations, adjust urgency, and close the response loop.</p></div><div className="header-actions"><button disabled={busy} onClick={() => void command("/api/v1/demo/advance", {}, "Advanced one deterministic sensor event.")}>Advance signal</button><button disabled={busy} onClick={() => void command("/api/v1/demo/reset", {}, "Demo reset to baseline.")}>Reset</button></div></header>
    <main className="crm-layout nurse-layout">
      <section className="panel queue-panel nurse-queue">
        <div className="panel-head"><div><span className="eyebrow">Re-triage queue</span><h2>{session.queue.length} patients</h2></div><span className="system-tag">Ranked live</span></div>
        <div className="nurse-list">{session.queue.map((patient, index) => <button className={keyFor(patient) === keyFor(selected ?? {}) ? "selected" : ""} onClick={() => setSelectedKey(keyFor(patient))} key={keyFor(patient)}><span className="rank">{String(index + 1).padStart(2, "0")}</span><span className="patient-name"><strong>{patient.name}</strong><small>Seat {patient.seat} / {patient.wait_minutes} min</small></span><span className={`esi esi-${patient.current_esi}`}>ESI {patient.current_esi}</span><span className="signal"><strong>{patient.latest_signal}</strong><small>{patient.status?.replace(/_/g, " ")}</small></span></button>)}</div>
      </section>
      <section className="panel clinical-record">
        <div className="patient-banner"><div><span className="eyebrow">Selected patient / Seat {selected?.seat}</span><h2>{selected?.name}</h2><p>{selected?.age} / {selected?.gender} / {selected?.visit}</p></div><span className={`esi-large esi-${selected?.current_esi}`}>ESI {selected?.current_esi}</span></div>
        <div className="clinical-grid">
          <div className="record-section"><span className="eyebrow">Latest vitals</span><div className="vital-grid">{vitals.length ? vitals.map(([label, vital]) => <div key={label}><small>{label.replace(/_/g, " ")}</small><strong>{vital.value}<i>{vital.unit}</i></strong></div>) : <p>No charted vitals.</p>}</div></div>
          <div className="record-section"><span className="eyebrow">Active conditions</span><ul>{selected?.chart?.conditions.length ? selected.chart.conditions.map((item) => <li key={item}>{item}</li>) : <li>None charted</li>}</ul></div>
          <div className="record-section"><span className="eyebrow">Risk context</span><ul>{selected?.risk_factors?.length ? selected.risk_factors.map((item) => <li key={item}>{item.replace(/_/g, " ")}</li>) : <li>No material risk factors</li>}</ul></div>
          <div className="record-section"><span className="eyebrow">Current signal</span><p>{selected?.latest_signal}</p><small>Baseline deviation {Math.round((selected?.baseline_deviation ?? 0) * 100)}%</small></div>
        </div>
        <div className={`clinical-alert ${alert ? "active" : ""}`}><div><span className="eyebrow">Response state</span><h3>{alert?.title ?? "No active escalation"}</h3><p>{alert?.evidence?.join(" / ") ?? "Continue scheduled reassessment."}</p></div><strong>{alert?.state.replace(/_/g, " ") ?? "Monitoring"}</strong></div>
        <div className="clinical-actions">
          {alert?.state === "page_pending" ? <button className="primary" disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/acknowledge`, { actor: "Charge RN" }, `Alert acknowledged for ${selected?.name}.`)}>Acknowledge alert</button> : null}
          {alert && ["acknowledged", "checkin"].includes(alert.state) ? <><button className="primary" disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/feedback`, { actor: "Charge RN", outcome: "confirmed" }, "Incident confirmed and response completed.")}>Confirm incident</button><button disabled={busy} onClick={() => void command(`/api/v1/alerts/${alert.alert_id}/feedback`, { actor: "Charge RN", outcome: "false_alarm" }, "Signal closed as a false alarm.")}>False alarm</button></> : null}
          {selected?.patient_id && (selected.current_esi ?? 5) > 1 ? <button disabled={busy} onClick={() => void command(`/api/v1/patients/${selected.patient_id}/esi`, { actor: "Charge RN", new_esi: Math.max(1, (selected.current_esi ?? 5) - 1) }, `Urgency raised to ESI ${Math.max(1, (selected.current_esi ?? 5) - 1)}.`)}>Raise urgency</button> : null}
        </div>
        <div className="note-entry"><label htmlFor="nursing-note">Nursing note</label><textarea id="nursing-note" value={note} onChange={(event) => setNote(event.target.value)} placeholder="Document reassessment or response..." /><button disabled={!note.trim()} onClick={() => { setNotice("Note staged in the demo record. Connect the clinical record integration to persist it."); setNote(""); }}>Add note</button></div>
        <p className="notice" aria-live="polite">{notice}</p>
      </section>
    </main>
  </div>;
}
