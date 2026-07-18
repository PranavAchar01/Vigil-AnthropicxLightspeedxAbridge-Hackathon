"use client";

import { useEffect, useRef, useState } from "react";
import BackendFeed from "./BackendFeed";
import { supabase } from "./lib/supabase";

// The backend runs on a laptop behind an ephemeral tunnel. It publishes its current
// public URL to Supabase (vigil_runtime); the page resolves it at load time and
// reconnects if it changes. NEXT_PUBLIC_VIGIL_URL is an optional static fallback.
const FALLBACK_BACKEND = process.env.NEXT_PUBLIC_VIGIL_URL || "";
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";

type Line = { id: number; cls: string; text: string };
type Caps = Record<string, boolean>;
type Patient = {
  name: string;
  age?: number;
  gender?: string;
  visit?: string;
  baseline_esi?: number;
  conditions?: string[];
  medications?: string[];
  vitals?: Record<string, string>;
  avatar?: string | null;
};

const cleanText = (value: unknown) =>
  String(value ?? "")
    .replace(/[\u2013\u2014]/g, " / ")
    .replace(/[\u2600-\u27BF]/g, "")
    .replace(/[\uD83C-\uDBFF][\uDC00-\uDFFF]/g, "")
    .replace(/\s+/g, " ")
    .trim();

function SectionTitle({ index, title, meta }: { index: string; title: string; meta?: string }) {
  return (
    <div className="section-title">
      <span className="section-index">{index}</span>
      <div>
        <h2>{title}</h2>
        {meta ? <p>{meta}</p> : null}
      </div>
    </div>
  );
}

export default function Home() {
  const [conn, setConn] = useState<"connecting" | "live" | "down">("connecting");
  const [patient, setPatient] = useState<Patient | null>(null);
  const [caps, setCaps] = useState<Caps>({});
  const [lines, setLines] = useState<Line[]>([
    { id: 0, cls: "system-line", text: "No event requires review." },
  ]);
  const [banner, setBanner] = useState<{ text: string; detail: string; alert: boolean }>({
    text: "Monitoring",
    detail: "No active escalation",
    alert: false,
  });
  const [logItems, setLogItems] = useState<string[]>([]);
  const [note, setNote] = useState<{ text: string; bundle?: string } | null>(null);
  const [camOk, setCamOk] = useState(false);
  const [backend, setBackend] = useState<string>(FALLBACK_BACKEND);

  const idRef = useRef(1);
  const deltaRef = useRef<number | null>(null);
  const freshRef = useRef(true);
  const traceEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    traceEnd.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [lines]);

  // Resolve the backend's current public URL from Supabase, and follow changes so a
  // tunnel restart reconnects the page with no redeploy.
  useEffect(() => {
    const sb = supabase;
    if (!sb) return;
    let active = true;
    const apply = (url?: string | null) => {
      if (active && typeof url === "string" && url) setBackend(url);
    };
    sb.from("vigil_runtime")
      .select("url")
      .eq("id", "backend")
      .maybeSingle()
      .then(({ data }) => apply((data as { url?: string } | null)?.url));
    const ch = sb
      .channel("vigil_runtime_backend")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "vigil_runtime", filter: "id=eq.backend" },
        (p) => apply((p.new as { url?: string }).url)
      )
      .subscribe();
    return () => {
      active = false;
      sb.removeChannel(ch);
    };
  }, []);

  useEffect(() => {
    if (!backend) return;
    setCamOk(false);
    const wsUrl = backend.replace(/^http/, "ws") + "/events";
    let ws: WebSocket | null = null;
    let closed = false;

    const pushLine = (cls: string, text: string) => {
      if (freshRef.current) {
        setLines([]);
        freshRef.current = false;
      }
      const id = idRef.current++;
      setLines((previous) => [...previous, { id, cls, text: cleanText(text) }]);
      return id;
    };

    const handle = (event: { type: string; payload: Record<string, unknown> }) => {
      const payload = event.payload;
      switch (event.type) {
        case "patient":
          setPatient(payload as unknown as Patient);
          break;
        case "status":
          if (payload.capabilities) setCaps(payload.capabilities as Caps);
          if (payload.level === "error") {
            setBanner({
              text: "Needs attention",
              detail: cleanText(payload.message),
              alert: true,
            });
            setLogItems((items) => [`Configuration: ${cleanText(payload.message)}`, ...items]);
          }
          break;
        case "perception":
          pushLine(
            "sensor-line",
            `${payload.modality}: ${payload.kind} / confidence ${payload.confidence}`,
          );
          break;
        case "fused":
          freshRef.current = true;
          deltaRef.current = null;
          pushLine(
            "signal-line",
            `${payload.summary} / ${String(payload.severity).toUpperCase()} / ${(payload.kinds as string[]).join(" + ")} / confidence ${payload.confidence}`,
          );
          break;
        case "reasoning_start":
          pushLine(
            "system-line",
            `Reviewing ${payload.patient} / prior ESI ${payload.prior_esi}`,
          );
          break;
        case "reasoning_delta": {
          const text = cleanText(payload.text);
          if (deltaRef.current === null) {
            deltaRef.current = pushLine("reason-line", text);
          } else {
            const target = deltaRef.current;
            setLines((previous) =>
              previous.map((line) =>
                line.id === target ? { ...line, text: cleanText(line.text + " " + text) } : line,
              ),
            );
          }
          break;
        }
        case "decision": {
          const changed = Number(payload.new_esi) < Number(payload.prior_esi);
          const escalate = Boolean(payload.escalate);
          deltaRef.current = null;
          setLines((previous) => [
            ...previous,
            {
              id: idRef.current++,
              cls: "decision-block",
              text: JSON.stringify({
                prior: payload.prior_esi,
                next: payload.new_esi,
                action: cleanText(payload.action),
                escalate,
                spoken: cleanText(payload.spoken_summary),
              }),
            },
          ]);
          if (changed || escalate) {
            setBanner({
              text: `ESI ${payload.new_esi}`,
              detail: cleanText(String(payload.action).replace(/_/g, " ")),
              alert: true,
            });
          }
          break;
        }
        case "call_status": {
          const status = cleanText(payload.status);
          if (status === "dialing") {
            setBanner({ text: "Calling nurse", detail: "Outbound call in progress", alert: true });
          } else if (status === "ringing") {
            setBanner({ text: "Nurse line ringing", detail: "Waiting for answer", alert: true });
          } else if (status === "failed") {
            setBanner({ text: "Call failed", detail: cleanText(payload.error), alert: true });
          } else if (status === "patient_checkin") {
            setBanner({ text: "Patient check-in", detail: "Voice assessment active", alert: true });
          }
          setLogItems((items) => [
            cleanText(`Call ${status}: ${payload.message ?? payload.conversation_id ?? payload.error ?? ""}`),
            ...items,
          ]);
          break;
        }
        case "escalation":
          setLogItems((items) => [
            cleanText(
              `${String(payload.kind).replace(/_/g, " ")}: ${payload.status} ${payload.message ?? ""}`,
            ),
            ...items,
          ]);
          break;
        case "note":
          setNote({
            text: cleanText(payload.text),
            bundle: cleanText(payload.bundle_path).split("/").pop(),
          });
          break;
      }
    };

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => setConn("live");
        ws.onmessage = (message) => {
          try {
            handle(JSON.parse(message.data));
          } catch {
            return;
          }
        };
        ws.onclose = () => {
          setConn("down");
          if (!closed) window.setTimeout(connect, 3000);
        };
        ws.onerror = () => ws?.close();
      } catch {
        setConn("down");
      }
    };

    connect();
    return () => {
      closed = true;
      ws?.close();
    };
  }, [backend]);

  const capability = (name: string, fallback = false) => Boolean(caps[name] || fallback);
  const patientMeta = patient
    ? [patient.age ? `${patient.age} years` : null, patient.gender, patient.visit]
        .filter(Boolean)
        .map(cleanText)
        .join(" / ")
    : "Recognition begins when a patient enters the camera view.";

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar surface">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <h1>Vigil</h1>
            <p>Continuous re-triage</p>
          </div>
        </div>

        <div className="session-details" aria-label="Session details">
          <span>Waiting room 01</span>
          <span>Skeleton view only</span>
        </div>

        <div className={`connection-state ${conn}`}>
          <span className="status-light" />
          <div>
            <strong>{conn === "live" ? "System live" : conn === "down" ? "Edge offline" : "Connecting"}</strong>
            <small>{conn === "live" ? "Signals are streaming" : "Local preview remains available"}</small>
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="workspace-panel observation-panel surface">
          <SectionTitle index="01" title="Live observation" meta="Camera and patient context" />

          <div className="video-frame inset-surface">
            <img
              src={`${backend}/video`}
              alt="Live waiting room pose feed"
              onLoad={(event) => {
                if ((event.target as HTMLImageElement).naturalWidth) setCamOk(true);
              }}
              onError={() => setCamOk(false)}
              style={{ display: camOk ? "block" : "none" }}
            />
            <div className="video-hud">
              <span>CAM 01</span>
              <span className={camOk ? "hud-live" : ""}>{camOk ? "Streaming" : "Standby"}</span>
            </div>
            {!camOk ? (
              <div className="camera-empty">
                <div className="scanner" aria-hidden="true">
                  <i />
                </div>
                <strong>Camera ready</strong>
                <span>Waiting for the edge feed</span>
              </div>
            ) : null}
          </div>

          <article className="patient-card inset-surface">
            <div className="patient-heading">
              <div>
                <span className="eyebrow">Patient context</span>
                <h3>{patient ? cleanText(patient.name) : "Waiting for patient"}</h3>
                <p>{patientMeta}</p>
              </div>
              <div className={`esi-tile ${patient ? "identified" : ""}`}>
                <span>Baseline</span>
                <strong>{patient ? `ESI ${patient.baseline_esi}` : "Pending"}</strong>
              </div>
            </div>

            <div className="patient-grid">
              <div className="vitals-block">
                <span className="field-label">Latest vitals</span>
                <div className="vitals-list">
                  {patient?.vitals && Object.keys(patient.vitals).length ? (
                    Object.entries(patient.vitals)
                      .slice(0, 4)
                      .map(([key, value]) => (
                        <div key={key}>
                          <span>{cleanText(key).replace(/_/g, " ")}</span>
                          <strong>{cleanText(value)}</strong>
                        </div>
                      ))
                  ) : (
                    <p className="quiet-copy">No chart selected</p>
                  )}
                </div>
              </div>

              <div className="clinical-block">
                <span className="field-label">Active conditions</span>
                <div className="chips">
                  {(patient?.conditions?.length ? patient.conditions : ["No data"]).slice(0, 5).map((item) => (
                    <span className="chip" key={item}>{cleanText(item)}</span>
                  ))}
                </div>
                <span className="field-label medications-label">Medications</span>
                <div className="chips">
                  {(patient?.medications?.length ? patient.medications : ["No data"]).slice(0, 5).map((item) => (
                    <span className="chip" key={item}>{cleanText(item)}</span>
                  ))}
                </div>
              </div>
            </div>
          </article>
        </section>

        <section className="workspace-panel reasoning-panel surface">
          <div className="panel-title-row">
            <SectionTitle index="02" title="Clinical reasoning" meta="Chart-grounded re-triage" />
            <span className="mode-pill"><i /> Agent ready</span>
          </div>

          <div className="reasoning-status inset-surface">
            <div className="pulse-graph" aria-hidden="true"><i /><i /><i /><i /><i /></div>
            <div>
              <span className="eyebrow">Current state</span>
              <strong>{banner.alert ? "Review in progress" : "Continuous monitoring"}</strong>
            </div>
            <span className="reasoning-time">Real time</span>
          </div>

          <div className="trace" aria-live="polite">
            <div className="trace-rail" aria-hidden="true" />
            {lines.map((line) =>
              line.cls === "decision-block" ? (
                <Decision key={line.id} raw={line.text} />
              ) : (
                <div key={line.id} className={`trace-line ${line.cls}`}>
                  <span className="trace-node" />
                  <p>{line.text}</p>
                </div>
              ),
            )}
            <div ref={traceEnd} />
          </div>
        </section>

        <section className="workspace-panel response-panel surface">
          <SectionTitle index="03" title="Response" meta="Escalation and record" />

          <div className={`response-state inset-surface ${banner.alert ? "alert" : ""}`}>
            <div className="response-orbit"><span /></div>
            <div>
              <span className="eyebrow">Escalation state</span>
              <h3>{banner.text}</h3>
              <p>{banner.detail}</p>
            </div>
          </div>

          <div className="capability-grid">
            <Capability label="Reasoning" active={capability("reasoning")} />
            <Capability label="Nurse call" active={capability("nurse_call")} />
            <Capability label="Patient check-in" active={capability("patient_checkin")} />
            <Capability label="Camera" active={camOk} />
          </div>

          <div className="response-section">
            <div className="subhead">
              <span>Activity</span>
              <small>{logItems.length ? `${logItems.length} updates` : "Quiet"}</small>
            </div>
            <div className="activity-list inset-surface">
              {logItems.length ? (
                logItems.slice(0, 6).map((item, index) => (
                  <div className="activity-item" key={`${item}-${index}`}>
                    <span />
                    <p>{item}</p>
                  </div>
                ))
              ) : (
                <div className="activity-empty">
                  <span />
                  <p>No escalations in this session</p>
                </div>
              )}
            </div>
          </div>

          <div className="response-section note-section">
            <div className="subhead">
              <span>Incident note</span>
              <small className={note?.bundle ? "record-ready" : ""}>{note?.bundle ? "FHIR ready" : "Awaiting incident"}</small>
            </div>
            <div className="note-preview inset-surface">
              {note?.text ? <p>{note.text}</p> : <p className="quiet-copy">The clinical note will appear after an escalation.</p>}
            </div>
          </div>

          <div className="mirror-state">
            <span className={SUPABASE_URL ? "online" : ""} />
            Event mirror {SUPABASE_URL ? "connected" : "not configured"}
          </div>
        </section>
      </main>

      <BackendFeed />
    </div>
  );
}

function Capability({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={`capability ${active ? "active" : ""}`}>
      <span />
      <div>
        <strong>{label}</strong>
        <small>{active ? "Ready" : "Standby"}</small>
      </div>
    </div>
  );
}

function Decision({ raw }: { raw: string }) {
  let decision: {
    prior?: number;
    next?: number;
    action?: string;
    escalate?: boolean;
    spoken?: string;
  } = {};
  try {
    decision = JSON.parse(raw);
  } catch {
    return null;
  }

  return (
    <article className={`decision-card ${decision.escalate ? "escalating" : ""}`}>
      <div className="decision-topline">
        <span className="trace-node" />
        <span>{decision.escalate ? "Escalation decision" : "Re-triage decision"}</span>
        <small>{cleanText(String(decision.action).replace(/_/g, " "))}</small>
      </div>
      <div className="esi-transition">
        <div><span>Prior</span><strong>ESI {decision.prior}</strong></div>
        <span className="transition-mark">to</span>
        <div><span>Current</span><strong>ESI {decision.next}</strong></div>
      </div>
      {decision.spoken ? (
        <div className="nurse-summary">
          <span>Nurse summary</span>
          <p>{cleanText(decision.spoken)}</p>
        </div>
      ) : null}
    </article>
  );
}
