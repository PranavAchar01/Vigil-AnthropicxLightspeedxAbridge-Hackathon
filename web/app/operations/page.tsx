"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import DashboardNav from "../DashboardNav";
import {
  fetchVigilSession,
  localSession,
  postVigilCommand,
  type QueuePatient,
  type VigilRole,
  type VigilSession,
} from "../demoSession";
import { useVigilBackend } from "../lib/useVigilBackend";

const OPERATIONS_ROLES: Array<{ value: VigilRole; label: string }> = [
  { value: "front_desk", label: "Front desk" },
  { value: "security", label: "Security" },
];

const patientKey = (patient: QueuePatient, index = 0) =>
  patient.patient_id ?? patient.patient_ref ?? patient.seat ?? `seat-${index}`;

export default function OperationsDashboard() {
  const backend = useVigilBackend();
  const [role, setRole] = useState<VigilRole>("front_desk");
  const [session, setSession] = useState<VigilSession>(() => localSession("front_desk", 0));
  const [selectedKey, setSelectedKey] = useState(() => patientKey(session.queue[0]));
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("No operations action is waiting.");

  const refresh = useCallback(async (nextRole: VigilRole) => {
    try {
      if (!backend) throw new Error("Backend unavailable");
      const next = await fetchVigilSession(backend, nextRole);
      setSession(next);
      return next;
    } catch {
      const next = localSession(nextRole, 0);
      setSession(next);
      return next;
    }
  }, [backend]);

  useEffect(() => {
    void refresh(role).then((next) => setSelectedKey(patientKey(next.queue[0])));
  }, [refresh, role]);

  useEffect(() => {
    const interval = window.setInterval(() => void refresh(role), 8000);
    return () => window.clearInterval(interval);
  }, [refresh, role]);

  const selected = session.queue.find((patient, index) => patientKey(patient, index) === selectedKey) ?? session.queue[0];
  const flaggedCount = session.queue.filter((patient) => patient.flagged || patient.medical_assist_needed).length;
  const longestWait = useMemo(
    () => Math.max(0, ...session.queue.map((patient) => patient.wait_minutes ?? 0)),
    [session.queue],
  );

  const requestAssist = async () => {
    if (!selected?.seat) return;
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, "/api/v1/operations/medical-assist", role, {
        actor: role === "security" ? "Security desk" : "Front desk",
        seat: selected.seat,
        reason: "Patient or companion requested clinical help",
      });
      if (!response.ok) throw new Error("Assist request unavailable");
      await refresh(role);
      setNotice(`Medical assist routed for seat ${selected.seat}. Charge nurse acknowledgement is now pending.`);
    } catch {
      setSession((current) => ({
        ...current,
        source: "local",
        queue: current.queue.map((patient, index) =>
          patientKey(patient, index) === selectedKey
            ? { ...patient, flagged: true, medical_assist_needed: true, alert_type: "medical assist" }
            : patient,
        ),
      }));
      setNotice(`Preview recorded a medical-assist request for seat ${selected.seat}.`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-shell operations-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <DashboardNav current="operations" />

      <header className="dashboard-header surface">
        <div>
          <span className="eyebrow">Waiting room 01</span>
          <h1>Operations</h1>
          <p>Route help by seat. Clinical details and camera data are not included in this view.</p>
        </div>
        <div className="dashboard-header-controls">
          <label className="role-control">
            <span>Viewing as</span>
            <select
              value={role}
              onChange={(event) => setRole(event.target.value as VigilRole)}
              aria-label="View operations as role"
            >
              {OPERATIONS_ROLES.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
            </select>
          </label>
          <span className={`source-pill ${session.source === "backend" ? "connected" : ""}`}>
            {session.source === "backend" ? "Backend connected" : "Local replay data"}
          </span>
        </div>
      </header>

      <section className="operations-metrics" aria-label="Waiting room summary">
        <article className="metric-card surface"><span>Occupied seats</span><strong>{session.queue.length}</strong><small>Monitored locations</small></article>
        <article className="metric-card surface"><span>Needs attention</span><strong>{flaggedCount}</strong><small>Seat-level flags</small></article>
        <article className="metric-card surface"><span>Longest wait</span><strong>{longestWait || "Restricted"}</strong><small>{longestWait ? "minutes" : "Not in this role scope"}</small></article>
        <article className="metric-card surface"><span>Role filter</span><strong>Active</strong><small>{session.scopes.length} scoped permissions</small></article>
      </section>

      <main className="operations-workspace">
        <section className="seat-board surface">
          <div className="panel-heading-row">
            <div><span className="eyebrow">Current occupancy</span><h2>Seat board</h2></div>
            <button type="button" onClick={() => void refresh(role)} disabled={busy}>Refresh</button>
          </div>
          <div className="seat-grid">
            {session.queue.map((patient, index) => {
              const key = patientKey(patient, index);
              const flagged = Boolean(patient.flagged || patient.medical_assist_needed);
              return (
                <button
                  type="button"
                  className={`seat-card ${key === selectedKey ? "selected" : ""} ${flagged ? "flagged" : ""}`}
                  onClick={() => setSelectedKey(key)}
                  aria-pressed={key === selectedKey}
                  key={key}
                >
                  <span className="seat-number">{patient.seat ?? "Zone"}</span>
                  <strong>{patient.name ?? patient.patient_ref ?? "Identity withheld"}</strong>
                  <small>{patient.wait_minutes != null ? `${patient.wait_minutes} min wait` : "Clinical identity withheld"}</small>
                  <i>{flagged ? "Help requested" : "Monitoring"}</i>
                </button>
              );
            })}
          </div>
        </section>

        <aside className="operations-action surface">
          <span className="eyebrow">Selected seat</span>
          <div className="selected-location">
            <strong>{selected?.seat ?? "No seat"}</strong>
            <div>
              <h2>{selected?.name ?? selected?.patient_ref ?? "Identity withheld"}</h2>
              <p>{selected?.medical_assist_needed || selected?.flagged ? "Clinical response requested" : "No active request"}</p>
            </div>
          </div>

          <div className="privacy-contract inset-surface">
            <span>Visible in this role</span>
            <div className="chips">
              {(role === "front_desk" ? ["Name", "Seat", "Wait time", "Flag state"] : ["Anonymous reference", "Seat", "Assist state"])
                .map((item) => <span className="chip" key={item}>{item}</span>)}
            </div>
            <p>Diagnosis, ESI, medications, agent reasoning, and video are removed by the API.</p>
          </div>

          <button
            className="assist-button"
            type="button"
            onClick={() => void requestAssist()}
            disabled={busy || !selected?.seat || selected?.medical_assist_needed || selected?.flagged}
          >
            {selected?.medical_assist_needed || selected?.flagged ? "Medical assist routed" : busy ? "Routing" : "Request medical assist"}
          </button>
          <p className="operations-notice" aria-live="polite">{notice}</p>
        </aside>
      </main>
    </div>
  );
}
