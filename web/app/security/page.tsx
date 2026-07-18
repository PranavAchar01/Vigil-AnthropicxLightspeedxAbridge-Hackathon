"use client";

import { useMemo, useState } from "react";
import DashboardNav from "../DashboardNav";
import { postVigilCommand } from "../demoSession";
import { useRoleSession } from "../useRoleSession";

const cameras = [
  { id: "CAM 01", zone: "Main waiting room", seats: ["A3", "B1"], state: "Online" },
  { id: "CAM 02", zone: "East entrance", seats: ["C2"], state: "Online" },
  { id: "CAM 03", zone: "Triage hallway", seats: [], state: "Online" },
  { id: "CAM 04", zone: "North exit", seats: [], state: "Online" },
];

export default function SecurityPage() {
  const { backend, session, refresh } = useRoleSession("security", 1800);
  const [cameraId, setCameraId] = useState("CAM 01");
  const [selectedSeat, setSelectedSeat] = useState("A3");
  const [notice, setNotice] = useState("Camera views show derived occupancy and safety events only.");
  const [busy, setBusy] = useState(false);
  const camera = cameras.find((item) => item.id === cameraId) ?? cameras[0];
  const occupants = useMemo(() => session.queue.filter((item) => camera.seats.includes(item.seat ?? "")), [camera.seats, session.queue]);
  const selected = session.queue.find((item) => item.seat === selectedSeat) ?? occupants[0];

  const flagForAssist = async () => {
    if (!selected?.seat) return;
    setBusy(true);
    try {
      if (!backend) throw new Error("Backend unavailable");
      const response = await postVigilCommand(backend, "/api/v1/operations/medical-assist", "security", { actor: `Security / ${camera.id}`, seat: selected.seat, reason: `Security safety flag from ${camera.zone}` });
      if (!response.ok) throw new Error(await response.text());
      setNotice(`Seat ${selected.seat} flagged. The charge nurse queue now shows a medical-assist request.`);
      await refresh();
    } catch {
      setNotice("Live flag unavailable. Connect the Vigil backend to route this request.");
    } finally { setBusy(false); }
  };

  return <div className="app-shell">
    <DashboardNav current="security" />
    <header className="crm-header panel"><div><span className="eyebrow">Security CRM</span><h1>Camera and safety operations</h1><p>Monitor zones, review anonymous safety flags, and route clinical help without opening patient charts.</p></div><div className="header-status"><span><i /> {cameras.filter((item) => item.state === "Online").length} cameras online</span><strong>{session.queue.filter((item) => item.medical_assist_needed).length} active flags</strong></div></header>
    <main className="security-layout">
      <section className="camera-selector panel">
        <div className="panel-head"><div><span className="eyebrow">Camera network</span><h2>Waiting room zones</h2></div><button onClick={() => void refresh()}>Refresh</button></div>
        <div className="camera-list">{cameras.map((item) => <button className={item.id === camera.id ? "selected" : ""} onClick={() => { setCameraId(item.id); if (item.seats[0]) setSelectedSeat(item.seats[0]); }} key={item.id}><span className="camera-index">{item.id.replace("CAM ", "")}</span><span><strong>{item.zone}</strong><small>{item.seats.length} tracked positions</small></span><i />{item.state}</button>)}</div>
        <div className="privacy-note"><span className="eyebrow">Access policy</span><p>Security receives anonymous references, seat locations, and medical-assist state. Names, diagnoses, ESI, and clinical reasoning are removed by the API.</p></div>
      </section>
      <section className="camera-console panel">
        <div className="camera-console-head"><div><span>{camera.id}</span><strong>{camera.zone}</strong></div><div><i /> {camera.state}</div></div>
        <div className="camera-field">
          <div className="field-grid" />
          <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
          {occupants.length ? occupants.map((person, index) => <button style={{ left: `${28 + index * 38}%`, top: `${40 + index * 15}%` }} className={`track-marker ${person.medical_assist_needed ? "flagged" : ""}`} onClick={() => setSelectedSeat(person.seat ?? "")} key={person.patient_ref ?? person.seat}><i /><strong>{person.seat}</strong><small>{person.medical_assist_needed ? "ASSIST" : "TRACKED"}</small></button>) : <div className="clear-zone"><strong>Zone clear</strong><span>No active tracked positions</span></div>}
        </div>
        <div className="console-data"><div><small>Mode</small><strong>Derived event view</strong></div><div><small>Retention</small><strong>No raw video stored</strong></div><div><small>Zone occupancy</small><strong>{occupants.length}</strong></div><div><small>Last sync</small><strong>Current</strong></div></div>
      </section>
      <aside className="security-response panel">
        <div className="panel-head"><div><span className="eyebrow">Safety response</span><h2>{selected ? `Seat ${selected.seat}` : "No track selected"}</h2></div>{selected?.medical_assist_needed ? <span className="system-tag urgent">Flagged</span> : null}</div>
        {selected ? <><dl className="record-list"><div><dt>Anonymous ref</dt><dd>{selected.patient_ref}</dd></div><div><dt>Location</dt><dd>{camera.zone} / {selected.seat}</dd></div><div><dt>Track state</dt><dd>Active</dd></div><div><dt>Clinical access</dt><dd>Restricted</dd></div><div><dt>Assist state</dt><dd>{selected.medical_assist_needed ? "Nurse notified" : "No active request"}</dd></div></dl><div className="assist-form"><span className="eyebrow">Create safety flag</span><p>Routes a nonclinical medical-assist request into the same live nurse queue.</p><button className="primary" disabled={busy || selected.medical_assist_needed} onClick={() => void flagForAssist()}>{selected.medical_assist_needed ? "Nurse already notified" : "Flag for medical assist"}</button></div></> : <p className="empty-message">Select a camera with an active track.</p>}
        <div className="flag-log"><span className="eyebrow">Current flags</span>{session.queue.filter((item) => item.medical_assist_needed).map((item) => <button onClick={() => setSelectedSeat(item.seat ?? "")} key={item.patient_ref}><span>{item.seat}</span><strong>Medical assist</strong><small>Nurse queue</small></button>)}</div>
        <p className="notice" aria-live="polite">{notice}</p>
      </aside>
    </main>
  </div>;
}
