"use client";

import { useEffect, useState } from "react";
import { supabase } from "./lib/supabase";

type Row = {
  id: number;
  created_at: string;
  type: string;
  source?: string | null;
  patient?: string | null;
  summary?: string | null;
};

const TYPE_LABEL: Record<string, string> = {
  perception: "sensor",
  fused: "event",
  decision: "re-triage",
  call_status: "call",
  escalation: "escalation",
  note: "note",
  tool_call: "agent",
  conversation_turn: "conversation",
};

function tone(type: string): string {
  if (type === "decision" || type === "escalation" || type === "call_status") return "alert";
  if (type === "fused" || type === "perception") return "signal";
  if (type === "tool_call" || type === "conversation_turn") return "agent";
  return "";
}

export default function BackendFeed() {
  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState<"off" | "connecting" | "live">(supabase ? "connecting" : "off");

  useEffect(() => {
    const sb = supabase;
    if (!sb) return;
    let active = true;
    sb.from("vigil_events")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(40)
      .then(({ data }) => { if (active && data) setRows(data as Row[]); });

    const ch = sb
      .channel("vigil_events_feed")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "vigil_events" }, (p) => {
        setRows((prev) => [p.new as Row, ...prev].slice(0, 150));
      })
      .subscribe((s) => { if (s === "SUBSCRIBED") setStatus("live"); });

    return () => { active = false; sb.removeChannel(ch); };
  }, []);

  return (
    <section className="panel glass feed">
      <div className="phead">
        <span className={`fdot ${status}`} />
        <span className="ptitle">Backend · live event stream</span>
        <span className="pill">Supabase {status === "live" ? "· live" : status === "off" ? "· not configured" : "· connecting"}</span>
      </div>
      <div className="feedbody">
        {rows.length === 0 ? (
          <div className="feedrow sysc">
            {status === "off"
              ? "Set NEXT_PUBLIC_SUPABASE_URL + key to stream the backend."
              : "Waiting for events — every sensor input, agent query, decision, and conversation turn lands here."}
          </div>
        ) : (
          rows.map((r) => (
            <div key={r.id} className={`feedrow ${tone(r.type)}`}>
              <span className="ftime">{new Date(r.created_at).toLocaleTimeString([], { hour12: false })}</span>
              <span className="fbadge">{TYPE_LABEL[r.type] ?? r.type}</span>
              <span className="fsrc">{r.source ?? ""}</span>
              <span className="fsum">{r.summary ?? r.type}</span>
              {r.patient ? <span className="fpat">{r.patient}</span> : null}
            </div>
          ))
        )}
      </div>
    </section>
  );
}
