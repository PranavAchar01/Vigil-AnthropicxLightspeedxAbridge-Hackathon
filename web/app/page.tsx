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
  name: string; age?: number; gender?: string; visit?: string;
  baseline_esi?: number; conditions?: string[]; medications?: string[];
  vitals?: Record<string, string>; avatar?: string | null;
};

const EyeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
const VideoIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="m16 13 5.22 3.48a.5.5 0 0 0 .78-.42V7.94a.5.5 0 0 0-.75-.43L16 10.5" />
    <rect x="2" y="6" width="14" height="12" rx="2" />
  </svg>
);
const PulseIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" />
  </svg>
);
const PhoneIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M13.83 16.57a1 1 0 0 0 1.21-.3l.36-.47A2 2 0 0 1 17 15h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2A18 18 0 0 1 2 4a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v3a2 2 0 0 1-.8 1.6l-.47.35a1 1 0 0 0-.29 1.23 14 14 0 0 0 6.39 6.38" />
  </svg>
);

export default function Home() {
  const [conn, setConn] = useState<"connecting" | "live" | "down">("connecting");
  const [patient, setPatient] = useState<Patient | null>(null);
  const [caps, setCaps] = useState<Caps>({});
  const [lines, setLines] = useState<Line[]>([{ id: 0, cls: "sysc", text: "Monitoring… waiting for a signal." }]);
  const [banner, setBanner] = useState<{ text: string; alert: boolean }>({ text: "Monitoring", alert: false });
  const [logItems, setLogItems] = useState<string[]>([]);
  const [note, setNote] = useState<{ text: string; bundle?: string } | null>(null);
  const [camOk, setCamOk] = useState(false);
  const [backend, setBackend] = useState<string>(FALLBACK_BACKEND);

  const idRef = useRef(1);
  const deltaRef = useRef<number | null>(null);
  const freshRef = useRef(true);
  const traceEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    traceEnd.current?.scrollIntoView({ block: "end" });
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
      setLines((prev) => [...prev, { id, cls, text }]);
      return id;
    };

    const handle = (ev: { type: string; payload: Record<string, unknown> }) => {
      const p = ev.payload as Record<string, unknown>;
      switch (ev.type) {
        case "patient":
          setPatient(p as unknown as Patient);
          break;
        case "status":
          if (p.capabilities) setCaps(p.capabilities as Caps);
          if (p.level === "error") {
            setBanner({ text: "Configuration needed", alert: true });
            setLogItems((l) => [`config · ${String(p.message ?? "")}`, ...l]);
          }
          break;
        case "perception":
          pushLine("faint", `· ${p.modality}: ${p.kind} (${p.confidence})`);
          break;
        case "fused":
          freshRef.current = true;
          deltaRef.current = null;
          pushLine("signal", `⚑ ${p.summary} — ${String(p.severity).toUpperCase()} [${(p.kinds as string[]).join(" + ")}] conf ${p.confidence}`);
          break;
        case "reasoning_start":
          pushLine("sysc", `Re-triaging ${p.patient} — prior ESI ${p.prior_esi}…`);
          break;
        case "reasoning_delta": {
          const t = String(p.text ?? "");
          if (deltaRef.current == null) {
            deltaRef.current = pushLine("reason", t);
          } else {
            const target = deltaRef.current;
            setLines((prev) => prev.map((ln) => (ln.id === target ? { ...ln, text: ln.text + t } : ln)));
          }
          break;
        }
        case "decision": {
          const changed = Number(p.new_esi) < Number(p.prior_esi);
          deltaRef.current = null;
          const esc = Boolean(p.escalate);
          setLines((prev) => [
            ...prev,
            {
              id: idRef.current++,
              cls: "decision-block",
              text: JSON.stringify({
                prior: p.prior_esi, next: p.new_esi, action: p.action, escalate: esc, spoken: p.spoken_summary,
              }),
            },
          ]);
          if (changed || esc)
            setBanner({ text: `ESI ${p.new_esi} · ${String(p.action).replace(/_/g, " ").toUpperCase()}`, alert: true });
          break;
        }
        case "call_status": {
          const s = String(p.status);
          if (s === "dialing") setBanner({ text: "Dialing charge nurse…", alert: true });
          else if (s === "ringing") setBanner({ text: "Nurse phone ringing", alert: true });
          else if (s === "failed") setBanner({ text: "Call failed", alert: true });
          else if (s === "patient_checkin") setBanner({ text: "Voice check-in", alert: true });
          setLogItems((l) => [`call · ${s} ${String(p.message ?? p.conversation_id ?? p.error ?? "")}`, ...l]);
          break;
        }
        case "escalation":
          setLogItems((l) => [`${String(p.kind).replace(/_/g, " ")} → ${p.status} ${String(p.message ?? "")}`, ...l]);
          break;
        case "note":
          setNote({ text: String(p.text ?? ""), bundle: String(p.bundle_path ?? "").split("/").pop() });
          break;
      }
    };

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => setConn("live");
        ws.onmessage = (e) => {
          try { handle(JSON.parse(e.data)); } catch { /* ignore */ }
        };
        ws.onclose = () => {
          setConn("down");
          if (!closed) setTimeout(connect, 3000);
        };
        ws.onerror = () => ws?.close();
      } catch {
        setConn("down");
      }
    };
    connect();
    return () => { closed = true; ws?.close(); };
  }, [backend]);

  const setCap = (k: string) => (caps[k] ? "cap on" : "cap");

  return (
    <div className="stage">
      <header className="topbar glass">
        <span className="glass-icon"><EyeIcon /></span>
        <span className="brand-name">Vigil</span>
        <span className="sub">command center · continuous re-triage · skeletons only</span>
        <span className="spacer" />
        <span className="backend">backend: {backend ? backend.replace(/^https?:\/\//, "") : "resolving…"}</span>
        <span className={`conn glass-sub ${conn === "live" ? "live" : conn === "down" ? "down" : ""}`}>
          <span className="dot" />
          {conn === "live" ? "Live" : conn === "down" ? "Backend offline" : "Connecting…"}
        </span>
      </header>

      <main className="grid">
        {/* LEFT — skeleton view + chart */}
        <section className="panel glass">
          <div className="phead"><span className="glass-icon"><VideoIcon /></span><span className="ptitle">Live skeleton view</span></div>
          <div className="videoframe glass-sub">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {backend && (
              <img src={`${backend}/video`} alt="" onLoad={(e) => { if ((e.target as HTMLImageElement).naturalWidth) setCamOk(true); }} onError={() => setCamOk(false)} style={{ display: camOk ? "block" : "none" }} />
            )}
            {!camOk && (
              <div className="cam-standby">
                <span className="glass-icon lg"><VideoIcon /></span>
                <span>{backend ? "Camera standby" : "Waiting for backend…"}</span>
              </div>
            )}
          </div>
          <div className="card glass-sub">
            <div className="pheadrow">
              {patient?.avatar ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img className="avatar" alt="" src={`${backend}${patient.avatar}`} />
              ) : null}
              <div className="pmeta">
                <div className="pname">{patient ? `${patient.name} · ${patient.age ?? "?"} · ${patient.gender ?? ""}` : "Awaiting patient recognition"}</div>
                <div className="pvisit">{patient ? (patient.visit ?? "") : "Step in front of the camera to identify a patient"}</div>
              </div>
              {patient ? <span className="esi-badge glass-sub">ESI {patient.baseline_esi}</span> : null}
            </div>
            <div className="row"><span className="k">Vitals</span><span className="v">{patient?.vitals ? Object.entries(patient.vitals).map(([k, v]) => `${k} ${v}`).join(" · ") : "—"}</span></div>
            <div className="label">Active conditions</div>
            <div className="chips">{(patient?.conditions ?? ["—"]).slice(0, 8).map((c, i) => <span key={i} className="chip">{c}</span>)}</div>
            <div className="label">Medications</div>
            <div className="chips">{(patient?.medications ?? ["—"]).slice(0, 8).map((c, i) => <span key={i} className="chip">{c}</span>)}</div>
          </div>
        </section>

        {/* MIDDLE — reasoning trace */}
        <section className="panel glass">
          <div className="phead"><span className="glass-icon"><PulseIcon /></span><span className="ptitle">Agent reasoning trace</span></div>
          <div className="trace">
            {lines.map((ln) =>
              ln.cls === "decision-block" ? (
                <Decision key={ln.id} raw={ln.text} />
              ) : (
                <div key={ln.id} className={`line ${ln.cls}`}>{ln.text}</div>
              )
            )}
            <div ref={traceEnd} />
          </div>
        </section>

        {/* RIGHT — escalation + backend */}
        <section className="panel glass">
          <div className="phead"><span className="glass-icon"><PhoneIcon /></span><span className="ptitle">Escalation</span></div>
          <div className={`banner glass-sub ${banner.alert ? "alert" : ""}`}>{banner.text}</div>
          <div className="caps">
            <span className={setCap("reasoning")}><span className="d" />reasoning</span>
            <span className={setCap("nurse_call")}><span className="d" />nurse call</span>
            <span className={setCap("patient_checkin")}><span className="d" />check-in</span>
            <span className={camOk ? "cap on" : "cap"}><span className="d" />camera</span>
          </div>
          <div className="log glass-sub">
            {logItems.length === 0 ? <div className="item">No escalations yet.</div> :
              logItems.slice(0, 12).map((it, i) => <div key={i} className="item">{it}</div>)}
          </div>
          <div className="phead" style={{ marginTop: 14 }}>
            <span className="ptitle">Ambient SOAP note</span>
            <span className="pill">{note?.bundle ? `FHIR ✓ ${note.bundle}` : "FHIR"}</span>
          </div>
          <div className="note glass-sub">{note?.text ?? "No incident yet."}</div>
          <div className="sbrow">
            <span className="d" style={{ background: SUPABASE_URL ? "var(--live)" : undefined }} />
            Supabase backend: {SUPABASE_URL ? "connected" : "awaiting project URL"}
          </div>
        </section>
      </main>

      <BackendFeed />
    </div>
  );
}

function Decision({ raw }: { raw: string }) {
  let d: { prior?: number; next?: number; action?: string; escalate?: boolean; spoken?: string } = {};
  try { d = JSON.parse(raw); } catch { /* ignore */ }
  return (
    <div className={`decision glass-sub ${d.escalate ? "esc" : ""}`}>
      <span className="esi">ESI {d.prior} <span className="arrow">→</span> <b>{d.next}</b></span>
      &nbsp;&nbsp;<span className="sysc">{String(d.action).replace(/_/g, " ")}</span>
      {d.escalate ? <span className="tag">ESCALATING</span> : null}
      <div className="spoken">“{d.spoken}”</div>
    </div>
  );
}
