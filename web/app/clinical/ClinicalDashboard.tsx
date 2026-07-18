"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AgentPipeline from "../AgentPipeline";
import BackendFeed from "../BackendFeed";
import DashboardNav from "../DashboardNav";
import {
  ROLES,
  fetchVigilSession,
  localSession,
  postVigilCommand,
  type QueuePatient,
  type VigilRole,
  type VigilSession,
} from "../demoSession";

const CLINICAL_ROLES = ROLES.filter((item) =>
  ["charge_nurse", "attending", "triage_nurse"].includes(item.value),
);

const BACKEND = process.env.NEXT_PUBLIC_VIGIL_URL || "http://localhost:8000";

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

export default function ClinicalDashboard() {
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
  const [role, setRole] = useState<VigilRole>("charge_nurse");
  const [session, setSession] = useState<VigilSession>(() => localSession("charge_nurse", 0));
  const [selectedKey, setSelectedKey] = useState("demo-vega");
  const [commandBusy, setCommandBusy] = useState(false);
  const [demoPlaying, setDemoPlaying] = useState(false);

  const idRef = useRef(1);
  const deltaRef = useRef<number | null>(null);
  const freshRef = useRef(true);
  const traceEnd = useRef<HTMLDivElement>(null);
  const demoStepRef = useRef(0);

  useEffect(() => {
    traceEnd.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [lines]);

  useEffect(() => {
    const wsUrl = BACKEND.replace(/^http/, "ws") + "/events";
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
  }, []);

  const refreshSession = useCallback(async (nextRole: VigilRole, fallbackStep?: number) => {
    try {
      const next = await fetchVigilSession(BACKEND, nextRole);
      demoStepRef.current = next.demo.step;
      setSession(next);
      setConn((current) => (current === "down" ? "live" : current));
      return next;
    } catch {
      const next = localSession(nextRole, fallbackStep ?? demoStepRef.current);
      setSession(next);
      return next;
    }
  }, []);

  useEffect(() => {
    void refreshSession(role);
  }, [refreshSession, role]);

  const applyDemoNarrative = useCallback((step: number) => {
    const narratives: Record<number, { line: string; banner: { text: string; detail: string; alert: boolean } }> = {
      0: {
        line: "Three patient baselines established. Queue ranking includes acuity, time, chart risk, and change from baseline.",
        banner: { text: "Monitoring", detail: "No active escalation", alert: false },
      },
      1: {
        line: "Maria Vega: posture declined from baseline for 74 seconds. Tier 0 starts voice and text check-in.",
        banner: { text: "Patient check-in", detail: "Maria Vega, seat A3", alert: true },
      },
      2: {
        line: "Labored breathing corroborates postural decline. Cardiac history and low oxygen saturation raise urgency from ESI 3 to ESI 2.",
        banner: { text: "ESI 2", detail: "Nurse acknowledgement requested", alert: true },
      },
      3: {
        line: "Jordan Park: low-confidence movement above baseline. Check-in only, no nurse page.",
        banner: { text: "Check-in active", detail: "Ambiguous signal held below page threshold", alert: true },
      },
      4: {
        line: "Idris Cole: companion moved rapidly toward staff. Hard safety signal routed to seat B1.",
        banner: { text: "Medical assist", detail: "Companion alarm at seat B1", alert: true },
      },
      5: {
        line: "Replay complete. All decisions remain available with exact input hashes in the audit chain.",
        banner: { text: "Replay complete", detail: "Review, acknowledge, or label each alert", alert: false },
      },
    };
    const narrative = narratives[step] ?? narratives[5];
    setLines((previous) => [
      ...previous.slice(-5),
      { id: idRef.current++, cls: step >= 2 && step <= 4 ? "signal-line" : "system-line", text: narrative.line },
    ]);
    setBanner(narrative.banner);
  }, []);

  const runDemoStep = useCallback(
    async (targetStep?: number) => {
      const nextStep = Math.min(targetStep ?? demoStepRef.current + 1, 5);
      setCommandBusy(true);
      try {
        const response = await postVigilCommand(BACKEND, "/api/v1/demo/advance", "charge_nurse");
        if (!response.ok) throw new Error("Demo advance unavailable");
        await refreshSession(role, nextStep);
      } catch {
        demoStepRef.current = nextStep;
        setSession(localSession(role, nextStep));
      } finally {
        demoStepRef.current = nextStep;
        applyDemoNarrative(nextStep);
        setCommandBusy(false);
      }
    },
    [applyDemoNarrative, refreshSession, role],
  );

  const resetDemo = useCallback(async () => {
    setCommandBusy(true);
    try {
      const response = await postVigilCommand(BACKEND, "/api/v1/demo/reset", "charge_nurse");
      if (!response.ok) throw new Error("Demo reset unavailable");
      await refreshSession(role, 0);
    } catch {
      setSession(localSession(role, 0));
    } finally {
      demoStepRef.current = 0;
      applyDemoNarrative(0);
      setSelectedKey("demo-vega");
      setCommandBusy(false);
    }
  }, [applyDemoNarrative, refreshSession, role]);

  const playDemo = useCallback(async () => {
    if (demoPlaying) return;
    setDemoPlaying(true);
    await resetDemo();
    for (let step = 1; step <= 5; step += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      await runDemoStep(step);
    }
    setDemoPlaying(false);
  }, [demoPlaying, resetDemo, runDemoStep]);

  const updateAlert = useCallback(
    async (outcome: "acknowledge" | "confirmed" | "false_alarm") => {
      const patient = session.queue.find((item) => (item.patient_id ?? item.patient_ref) === selectedKey);
      const alert = patient?.alert;
      if (!alert) return;
      setCommandBusy(true);
      try {
        const path = outcome === "acknowledge"
          ? `/api/v1/alerts/${alert.alert_id}/acknowledge`
          : `/api/v1/alerts/${alert.alert_id}/feedback`;
        const body = outcome === "acknowledge"
          ? { actor: "Charge RN" }
          : { actor: "Charge RN", outcome };
        const response = await postVigilCommand(BACKEND, path, role, body);
        if (!response.ok) throw new Error("Alert update unavailable");
        await refreshSession(role);
      } catch {
        setSession((current) => ({
          ...current,
          queue: current.queue.map((item) => {
            if ((item.patient_id ?? item.patient_ref) !== selectedKey || !item.alert) return item;
            return {
              ...item,
              status: outcome === "acknowledge" ? "acknowledged" : "resolved",
              alert: {
                ...item.alert,
                state: outcome === "acknowledge" ? "acknowledged" : "resolved",
                acknowledged_by: outcome === "acknowledge" ? "Charge RN" : item.alert.acknowledged_by,
              },
            };
          }),
        }));
      } finally {
        setLogItems((items) => [
          outcome === "acknowledge" ? "Charge RN acknowledged the alert" : `Alert feedback recorded: ${outcome.replace("_", " ")}`,
          ...items,
        ]);
        setCommandBusy(false);
      }
    },
    [refreshSession, role, selectedKey, session.queue],
  );

  const capability = (name: string, fallback = false) => Boolean(caps[name] || fallback);
  const livePerception = capability("live_perception", false);
  const selectedPatient =
    session.queue.find((item) => (item.patient_id ?? item.patient_ref) === selectedKey) ??
    session.queue[0];
  const clinicalPatient = selectedPatient?.name && !selectedPatient.redacted ? selectedPatient : null;
  const roleHasClinicalContext = !["front_desk", "security", "compliance"].includes(role);
  const displayPatient: Patient | null = clinicalPatient
    ? {
        name: clinicalPatient.name ?? "Unknown patient",
        age: clinicalPatient.age,
        gender: clinicalPatient.gender,
        visit: clinicalPatient.visit,
        baseline_esi: clinicalPatient.current_esi,
        conditions: clinicalPatient.chart?.conditions,
        medications: clinicalPatient.chart?.medications,
        vitals: clinicalPatient.chart
          ? Object.fromEntries(
              Object.entries(clinicalPatient.chart.vitals).map(([key, value]) => [
                key,
                `${value.value}${value.unit}`,
              ]),
            )
          : undefined,
      }
    : roleHasClinicalContext
      ? patient
      : null;
  const patientMeta = displayPatient
    ? [displayPatient.age ? `${displayPatient.age} years` : null, displayPatient.gender, displayPatient.visit]
        .filter(Boolean)
        .map(cleanText)
        .join(" / ")
    : "Recognition begins when a patient enters the camera view.";
  const canViewObservation = session.scopes.includes("video:view") || role === "attending";
  const selectedAlert = selectedPatient?.alert;
  const limitedRole = role === "front_desk" || role === "security" || role === "compliance";
  const responseActive = Boolean(
    selectedAlert || selectedPatient?.flagged || selectedPatient?.medical_assist_needed || (!limitedRole && banner.alert),
  );
  const responseTitle = selectedAlert
    ? cleanText(selectedAlert.title)
    : selectedPatient?.medical_assist_needed
      ? "Medical assist needed"
      : selectedPatient?.flagged
        ? "Flagged for clinical review"
        : limitedRole
          ? "Monitoring"
          : banner.text;
  const responseDetail = selectedAlert
    ? `${cleanText(selectedAlert.state).replace(/_/g, " ")} / ESI ${selectedAlert.prior_esi} to ${selectedAlert.current_esi}`
    : selectedPatient?.medical_assist_needed
      ? `Seat ${selectedPatient.seat ?? "assigned zone"} / no clinical detail provided`
      : selectedPatient?.flagged
        ? "Clinical details restricted by scope"
        : limitedRole
          ? "No scoped alert for this selection"
          : banner.detail;
  const visibleLogItems = limitedRole ? [] : logItems;

  return (
    <div className="app-shell clinical-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <DashboardNav current="clinical" />

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
          <label className="role-control">
            <span>Viewing as</span>
            <select
              value={role}
              onChange={(event) => {
                const nextRole = event.target.value as VigilRole;
                setRole(nextRole);
                setSession(localSession(nextRole, demoStepRef.current));
                setSelectedKey("demo-vega");
              }}
              aria-label="View command center as role"
            >
              {CLINICAL_ROLES.map((item) => (
                <option value={item.value} key={item.value}>{item.label}</option>
              ))}
            </select>
          </label>
        </div>

        <div className={`connection-state ${conn}`}>
          <span className="status-light" />
          <div>
            <strong>{conn === "live" ? "System live" : conn === "down" ? "Edge offline" : "Connecting"}</strong>
            <small>{conn === "live" ? "Signals are streaming" : "Local preview remains available"}</small>
          </div>
        </div>
      </header>

      <AgentPipeline capabilities={caps} source={session.source} compact />

      <section className="queue-board surface" aria-label="Ranked re-triage queue">
        <div className="queue-heading">
          <div>
            <span className="eyebrow">Live priority model</span>
            <h2>Re-triage queue</h2>
            <p>Ranked by acuity, change from baseline, reassessment time, and chart risk</p>
          </div>
          <div className="demo-controls">
            <div className="demo-progress" aria-label={`Demo step ${session.demo.step} of ${session.demo.total_steps}`}>
              <span style={{ width: `${(session.demo.step / session.demo.total_steps) * 100}%` }} />
            </div>
            <span className="source-pill">{session.source === "backend" ? "Edge state" : "Stage-safe preview"}</span>
            <button data-testid="reset-demo" type="button" onClick={() => void resetDemo()} disabled={commandBusy}>Reset</button>
            <button data-testid="next-signal" type="button" onClick={() => void runDemoStep()} disabled={commandBusy || session.demo.step >= 5}>Next signal</button>
            <button data-testid="run-replay" className="primary" type="button" onClick={() => void playDemo()} disabled={commandBusy || demoPlaying}>
              {demoPlaying ? "Replaying" : "Run replay"}
            </button>
          </div>
        </div>

        <div className="queue-list">
          {session.queue.map((item, index) => {
            const key = item.patient_id ?? item.patient_ref ?? `queue-${index}`;
            return (
              <QueueCard
                patient={item}
                selected={key === selectedKey || (!session.queue.some((candidate) => (candidate.patient_id ?? candidate.patient_ref) === selectedKey) && index === 0)}
                onSelect={() => setSelectedKey(key)}
                key={key}
              />
            );
          })}
        </div>
      </section>

      <main className="workspace">
        <section className="workspace-panel observation-panel surface">
          <SectionTitle index="01" title="Live observation" meta="Camera and patient context" />

          <div className="video-frame inset-surface">
            {livePerception && canViewObservation ? (
              <img
                src={`${BACKEND}/video`}
                alt="Live waiting room pose feed"
                onLoad={(event) => {
                  if ((event.target as HTMLImageElement).naturalWidth) setCamOk(true);
                }}
                onError={() => setCamOk(false)}
                style={{ display: camOk ? "block" : "none" }}
              />
            ) : null}
            <div className="video-hud">
              <span>CAM 01</span>
              <span className={camOk && canViewObservation && livePerception ? "hud-live" : ""}>
                {!canViewObservation ? "Restricted" : livePerception ? (camOk ? "Streaming" : "Standby") : "Replay"}
              </span>
            </div>
            {!camOk || !canViewObservation || !livePerception ? (
              <div className="camera-empty">
                <div className="scanner" aria-hidden="true">
                  <i />
                </div>
                <strong>
                  {!canViewObservation ? "Video omitted by role scope" : livePerception ? "Camera ready" : "Replay ready"}
                </strong>
                <span>
                  {!canViewObservation
                    ? "The server did not send video data"
                    : livePerception
                      ? "Waiting for the edge feed"
                      : "Live perception is disabled for this deployment"}
                </span>
              </div>
            ) : null}
          </div>

          <article className="patient-card inset-surface">
            {clinicalPatient || displayPatient ? (
              <>
                <div className="patient-heading">
                  <div>
                    <span className="eyebrow">Patient context</span>
                    <h3>{displayPatient ? cleanText(displayPatient.name) : "Waiting for patient"}</h3>
                    <p>{patientMeta}</p>
                  </div>
                  <div className={`esi-tile ${displayPatient ? "identified" : ""}`}>
                    <span>{clinicalPatient?.initial_esi !== clinicalPatient?.current_esi ? "Re-triaged" : "Current"}</span>
                    <strong>{displayPatient ? `ESI ${displayPatient.baseline_esi}` : "Pending"}</strong>
                  </div>
                </div>

                <div className="patient-grid">
                  <div className="vitals-block">
                    <span className="field-label">Latest vitals</span>
                    <div className="vitals-list">
                      {displayPatient?.vitals && Object.keys(displayPatient.vitals).length ? (
                        Object.entries(displayPatient.vitals)
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
                      {(displayPatient?.conditions?.length ? displayPatient.conditions : ["No data"]).slice(0, 5).map((item) => (
                        <span className="chip" key={item}>{cleanText(item)}</span>
                      ))}
                    </div>
                    <span className="field-label medications-label">Medications</span>
                    <div className="chips">
                      {(displayPatient?.medications?.length ? displayPatient.medications : ["No data"]).slice(0, 5).map((item) => (
                        <span className="chip" key={item}>{cleanText(item)}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="redaction-card">
                <span className="redaction-mark" aria-hidden="true" />
                <span className="eyebrow">Minimum necessary access</span>
                <h3>Clinical context withheld</h3>
                <p>This role receives a server-filtered response. Names, chart data, reasoning, and video fields never cross the API boundary.</p>
                <code>{selectedPatient?.patient_ref ?? selectedPatient?.name ?? "restricted_by_scope"}</code>
              </div>
            )}
          </article>
        </section>

        <section className="workspace-panel reasoning-panel surface">
          <div className="panel-title-row">
            <SectionTitle index="02" title="Clinical reasoning" meta="Chart-grounded re-triage" />
            <span className="mode-pill"><i /> Agent ready</span>
          </div>

          {session.scopes.includes("reason:read") ? (
            <>
              <div className="reasoning-status inset-surface">
                <div className="pulse-graph" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                <div>
                  <span className="eyebrow">Current state</span>
                  <strong>{banner.alert ? "Review in progress" : "Continuous monitoring"}</strong>
                </div>
                <span className="reasoning-time">Tier 0 always on</span>
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
            </>
          ) : (
            <div className="scope-boundary inset-surface">
              <span className="scope-lock" aria-hidden="true" />
              <span className="eyebrow">Server response</span>
              <h3>Reasoning restricted by scope</h3>
              <p>The response for {ROLES.find((item) => item.value === role)?.label} contains no rationale or evidence fields.</p>
              <code>reason:read / denied</code>
            </div>
          )}
        </section>

        <section className="workspace-panel response-panel surface">
          <SectionTitle index="03" title="Response" meta="Escalation and record" />

          <div className={`response-state inset-surface ${responseActive ? "alert" : ""}`}>
            <div className="response-orbit"><span /></div>
            <div>
              <span className="eyebrow">Escalation state</span>
              <h3>{responseTitle}</h3>
              <p>{responseDetail}</p>
            </div>
          </div>

          <div className="capability-grid">
            <Capability label="Tier 0 guard" active={capability("multi_patient", true)} />
            <Capability label="Acknowledge" active={session.scopes.includes("escalate:ack")} />
            <Capability label="Text check-in" active={capability("demo_replay", true)} />
            <Capability label="Audit chain" active={session.audit_verified.valid} />
          </div>

          {selectedAlert && session.scopes.includes("escalate:ack") ? (
            <div className="alert-actions" aria-label="Alert actions">
              <button type="button" onClick={() => void updateAlert("acknowledge")} disabled={commandBusy || selectedAlert.state === "acknowledged"}>
                {selectedAlert.state === "acknowledged" ? "Acknowledged" : "Acknowledge"}
              </button>
              <button type="button" onClick={() => void updateAlert("confirmed")} disabled={commandBusy}>Confirmed</button>
              <button type="button" onClick={() => void updateAlert("false_alarm")} disabled={commandBusy}>False alarm</button>
            </div>
          ) : null}

          <div className="response-section">
            <div className="subhead">
              <span>Activity</span>
              <small>{visibleLogItems.length ? `${visibleLogItems.length} updates` : "Quiet"}</small>
            </div>
            <div className="activity-list inset-surface">
              {visibleLogItems.length ? (
                visibleLogItems.slice(0, 6).map((item, index) => (
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
              {note?.text ? (
                <p>{note.text}</p>
              ) : selectedAlert && session.scopes.includes("chart:read") ? (
                <p>
                  Vigil detected {cleanText(selectedAlert.title).toLowerCase()}. Acuity changed from ESI {selectedAlert.prior_esi} to ESI {selectedAlert.current_esi}. The nurse acknowledgement and source observations are linked in the FHIR Provenance record.
                </p>
              ) : (
                <p className="quiet-copy">The clinical note will appear after an escalation.</p>
              )}
            </div>
          </div>

          {session.audit ? (
            <div className="audit-section">
              <div className="subhead">
                <span>Audit chain</span>
                <small className={session.audit_verified.valid ? "record-ready" : ""}>
                  {session.audit_verified.valid ? `${session.audit_verified.blocks} blocks verified` : "Verification failed"}
                </small>
              </div>
              <div className="audit-list inset-surface">
                {session.audit.slice(-4).reverse().map((block) => (
                  <div className={`audit-row ${block.outcome === "denied" ? "denied" : ""}`} key={block.audit_id}>
                    <code>{block.hash.slice(0, 8)}</code>
                    <span>{cleanText(block.action).replace(/_/g, " ")}</span>
                    <small>{cleanText(block.outcome)}</small>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="mirror-state">
            <span className={session.audit_verified.valid ? "online" : ""} />
            Alert budget {session.alert_budget.used} of {session.alert_budget.limit} this shift
          </div>
        </section>
      </main>

      <BackendFeed audit={session.audit} />
    </div>
  );
}

function QueueCard({
  patient,
  selected,
  onSelect,
}: {
  patient: QueuePatient;
  selected: boolean;
  onSelect: () => void;
}) {
  const label = patient.name ?? patient.patient_ref ?? "Unidentified track";
  const active = Boolean(
    patient.alert || patient.flagged || patient.medical_assist_needed,
  );
  const state = patient.alert?.state ?? patient.status ?? (active ? "flagged" : "monitoring");
  return (
    <button
      type="button"
      className={`queue-card ${selected ? "selected" : ""} ${active ? "active" : ""}`}
      onClick={onSelect}
      aria-pressed={selected}
    >
      <span className="queue-rank">{patient.priority ? Math.round(patient.priority) : patient.seat ?? "ID"}</span>
      <span className="queue-person">
        <strong>{cleanText(label)}</strong>
        <small>
          {[patient.seat ? `Seat ${patient.seat}` : null, patient.wait_minutes != null ? `${patient.wait_minutes} min` : null]
            .filter(Boolean)
            .join(" / ") || "Identity withheld"}
        </small>
      </span>
      {patient.current_esi ? (
        <span className="queue-esi">
          <small>ESI</small>
          <strong>{patient.current_esi}</strong>
        </span>
      ) : null}
      <span className={`queue-state ${active ? "active" : ""}`}>
        <i />
        {cleanText(state).replace(/_/g, " ")}
      </span>
    </button>
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
