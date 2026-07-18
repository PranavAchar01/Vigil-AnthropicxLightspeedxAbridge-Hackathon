"use client";

import { useMemo, useState } from "react";
import DashboardNav from "../DashboardNav";
import { postVigilCommand, type QueuePatient } from "../demoSession";
import { useRoleSession } from "../useRoleSession";

export default function FrontDeskPage() {
  const { backend, session, refresh } = useRoleSession("front_desk");
  const [selectedSeat, setSelectedSeat] = useState("A3");
  const [reason, setReason] = useState("Patient or companion requested clinical help");
  const [notice, setNotice] = useState("Select a patient to review their nonclinical check-in status.");
  const [busy, setBusy] = useState(false);
  const selected = useMemo(() => session.queue.find((item) => item.seat === selectedSeat) ?? session.queue[0], [selectedSeat, session.queue]);
  const waiting = session.queue.filter((item) => !item.flagged).length;

  const requestAssist = async () => {
    if (!selected?.seat) return;
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, "/api/v1/operations/medical-assist", "front_desk", { actor: "Front desk", seat: selected.seat, reason });
      if (!response.ok) throw new Error(await response.text());
      setNotice(`Medical assist routed for seat ${selected.seat}. The nurse queue has been updated.`);
      await refresh();
    } catch {
      setNotice("Live request unavailable. Connect the Vigil backend to route this action.");
    } finally {
      setBusy(false);
    }
  };

  return <div className="app-shell">
    <DashboardNav current="front-desk" />
    <header className="crm-header panel"><div><span className="eyebrow">Front desk CRM</span><h1>Waiting room intake</h1><p>Check-in, seating, wait status, and clinical-assist routing without chart access.</p></div><div className="header-status"><span><i /> {session.source === "backend" ? "Live backend" : "Preview data"}</span><strong>{session.queue.length} checked in</strong></div></header>
    <main className="crm-layout front-layout">
      <section className="panel queue-panel">
        <div className="panel-head"><div><span className="eyebrow">Patient queue</span><h2>Waiting room</h2></div><button onClick={() => void refresh()}>Refresh</button></div>
        <div className="queue-summary"><div><strong>{session.queue.length}</strong><small>checked in</small></div><div><strong>{waiting}</strong><small>waiting</small></div><div><strong>{session.queue.length - waiting}</strong><small>assist flags</small></div></div>
        <div className="crm-list-head"><span>Patient</span><span>Seat</span><span>Wait</span><span>Status</span></div>
        <div className="crm-list">{session.queue.map((patient) => <button className={patient.seat === selected?.seat ? "selected" : ""} onClick={() => setSelectedSeat(patient.seat ?? "")} key={patient.patient_id ?? patient.seat}><span><strong>{patient.name}</strong><small>Verified check-in</small></span><span>{patient.seat}</span><span>{patient.wait_minutes} min</span><span className={patient.flagged ? "status urgent" : "status"}>{patient.flagged ? "Assist requested" : "Waiting"}</span></button>)}</div>
      </section>
      <aside className="panel detail-panel">
        <div className="panel-head"><div><span className="eyebrow">Selected patient</span><h2>{selected?.name ?? "No selection"}</h2></div><span className="seat-block">{selected?.seat}</span></div>
        <dl className="record-list"><div><dt>Check-in</dt><dd>Identity verified</dd></div><div><dt>Consent</dt><dd>Continuous monitoring accepted</dd></div><div><dt>Language</dt><dd>On file with clinical team</dd></div><div><dt>Wait time</dt><dd>{selected?.wait_minutes ?? 0} minutes</dd></div><div><dt>Clinical chart</dt><dd>Restricted for this role</dd></div></dl>
        <div className="assist-form"><span className="eyebrow">Clinical assist</span><p>Send a seat-level request directly to the charge nurse. No diagnosis or chart data is exposed here.</p><label htmlFor="assist-reason">Reason</label><select id="assist-reason" value={reason} onChange={(event) => setReason(event.target.value)}><option>Patient or companion requested clinical help</option><option>Patient appears to be getting worse</option><option>Mobility assistance requested</option><option>Communication support requested</option></select><button className="primary" disabled={busy || !selected} onClick={() => void requestAssist()}>{busy ? "Routing request" : `Request medical assist / ${selected?.seat ?? "seat"}`}</button></div>
        <p className="notice" aria-live="polite">{notice}</p>
      </aside>
    </main>
  </div>;
}
