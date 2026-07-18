"use client";

import { useEffect, useRef, useState } from "react";

type IntakeResult = {
  esi: number;
  esi_decision_point: string;
  esi_criteria: string;
  chief_complaint: string;
  predicted_resources: string[];
  red_flags: string[];
  confidence: number;
  rationale: string;
  spoken_summary: string;
  transcript: string;
};

type PatientOption = { id: string; name: string };

const ESI_TONE: Record<number, string> = { 1: "esi-1", 2: "esi-2", 3: "esi-3", 4: "esi-4", 5: "esi-5" };

// Voice intake -> initial ESI. Records in the browser, posts the audio (or typed
// text) to the backend's /intake endpoint, and renders the graded ESI. When a
// patient is selected the backend also sets that chart's baseline so the live
// re-triage loop escalates from this grade.
export default function VoiceIntake({ backend }: { backend: string }) {
  const [patients, setPatients] = useState<PatientOption[]>([]);
  const [patientId, setPatientId] = useState("");
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<IntakeResult | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (!backend) return;
    let active = true;
    fetch(`${backend}/patients`)
      .then((r) => r.json())
      .then((d: { patients?: Array<{ id: string; name: string }> }) => {
        if (active) setPatients((d.patients ?? []).map((p) => ({ id: p.id, name: p.name })));
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [backend]);

  const send = async (form: FormData) => {
    setBusy(true);
    setErr("");
    try {
      const r = await fetch(`${backend}/intake`, { method: "POST", body: form });
      const d = await r.json();
      if (!r.ok) throw new Error((d as { detail?: string }).detail || `HTTP ${r.status}`);
      setResult(d as IntakeResult);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Intake failed");
    } finally {
      setBusy(false);
    }
  };

  const toggleRecord = async () => {
    setErr("");
    if (recording) {
      recRef.current?.stop();
      return;
    }
    if (!backend) {
      setErr("Backend offline — cannot reach the grader.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      recRef.current = rec;
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("wav") ? "wav" : "webm";
        const fd = new FormData();
        fd.append("audio", blob, `intake.${ext}`);
        fd.append("patient_id", patientId);
        send(fd);
      };
      rec.start();
      setRecording(true);
    } catch (e) {
      setErr(`Microphone blocked: ${e instanceof Error ? e.message : "no access"}`);
    }
  };

  const gradeText = () => {
    if (!text.trim()) {
      setErr("Type an intake or use the mic.");
      return;
    }
    const fd = new FormData();
    fd.append("text", text.trim());
    fd.append("patient_id", patientId);
    send(fd);
  };

  return (
    <article className="patient-card inset-surface intake-card">
      <div className="intake-head">
        <div>
          <span className="eyebrow">Voice intake</span>
          <h3>Initial triage</h3>
        </div>
        <select
          className="intake-select"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          aria-label="Ground the intake in a patient chart"
        >
          <option value="">No chart — symptoms only</option>
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="intake-controls">
        <button
          type="button"
          className={`mic-btn ${recording ? "recording" : ""}`}
          onClick={toggleRecord}
          disabled={busy}
          aria-label="Record intake"
        >
          <span className="mic-glyph" aria-hidden="true" />
          {recording ? "Stop" : busy ? "Grading…" : "Record intake"}
        </button>
        <div className="intake-text">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && gradeText()}
            placeholder="or type symptoms…"
            aria-label="Type intake"
          />
          <button type="button" onClick={gradeText} disabled={busy}>
            Grade
          </button>
        </div>
      </div>

      {err ? <p className="intake-error">{err}</p> : null}

      {result ? (
        <div className="intake-result">
          <div className={`intake-esi ${ESI_TONE[result.esi] ?? ""}`}>
            <strong>{result.esi}</strong>
            <div>
              <span className="eyebrow">Initial ESI</span>
              <small>Decision {result.esi_decision_point}</small>
            </div>
          </div>
          <p className="intake-criteria">{result.esi_criteria}</p>
          {result.predicted_resources?.length ? (
            <div className="chips">
              {result.predicted_resources.slice(0, 5).map((r) => (
                <span className="chip" key={r}>
                  {r}
                </span>
              ))}
            </div>
          ) : null}
          {result.transcript ? <p className="intake-heard">Heard: “{result.transcript}”</p> : null}
          <p className="intake-confirm">Decision support — a clinician confirms this grade.</p>
        </div>
      ) : null}
    </article>
  );
}
