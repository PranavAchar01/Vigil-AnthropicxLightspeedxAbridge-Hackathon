"use client";

import { useEffect, useState } from "react";
import { supabase } from "./lib/supabase";
import type { AuditView } from "./demoSession";

type Row = {
  id: number | string;
  created_at: string;
  type: string;
  source?: string | null;
  patient?: string | null;
  summary?: string | null;
};

const TYPE_LABEL: Record<string, string> = {
  perception: "Sensor",
  fused: "Observation",
  decision: "Triage",
  call_status: "Call",
  escalation: "Escalation",
  note: "Note",
  tool_call: "Agent",
  conversation_turn: "Conversation",
};

function tone(type: string): string {
  if (type === "decision" || type === "escalation" || type === "call_status") return "alert";
  if (type === "fused" || type === "perception") return "signal";
  if (type === "tool_call" || type === "conversation_turn") return "agent";
  return "";
}

function cleanText(value: string | null | undefined): string {
  return (value ?? "")
    .replace(/[\u2013\u2014]/g, " / ")
    .replace(/[\u2700-\u27BF]/g, "")
    .replace(/[\u{1F300}-\u{1FAFF}]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}

function statusLabel(status: "off" | "connecting" | "live"): string {
  if (status === "live") return "Live";
  if (status === "connecting") return "Connecting";
  return "Not configured";
}

function eventTime(value: string): string {
  return new Date(value).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
  });
}

export default function BackendFeed({ audit }: { audit?: AuditView[] }) {
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
      .then(({ data }) => {
        if (active && data) setRows(data as Row[]);
      });

    const channel = sb
      .channel("vigil_events_feed")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "vigil_events" }, (payload) => {
        setRows((previous) => [payload.new as Row, ...previous].slice(0, 150));
      })
      .subscribe((subscriptionStatus) => {
        if (subscriptionStatus === "SUBSCRIBED") setStatus("live");
      });

    return () => {
      active = false;
      sb.removeChannel(channel);
    };
  }, []);

  const displayRows: Row[] = rows.length
    ? rows
    : (audit ?? []).slice().reverse().map((block) => ({
        id: block.audit_id,
        created_at: new Date(block.ts * 1000).toISOString(),
        type: block.action === "retriage_decision" ? "decision" : block.outcome === "denied" ? "escalation" : "tool_call",
        source: block.role,
        patient: block.resource.startsWith("patient:") ? block.resource.slice(8) : null,
        summary: `${block.action.replace(/_/g, " ")}: ${block.outcome} / hash ${block.hash.slice(0, 8)}`,
      }));

  return (
    <section className="event-dock surface" aria-label="System events">
      <div className="event-heading">
        <div className="event-title">
          <span className="section-index">04</span>
          <div>
            <h2>System events</h2>
            <p>Sensor and agent activity</p>
          </div>
        </div>
        <span className={`event-status ${status}`}>
          <i aria-hidden="true" />
          {rows.length ? statusLabel(status) : displayRows.length ? "Audit chain" : statusLabel(status)}
        </span>
      </div>

      <div className="event-table" role="table" aria-label="Recent Vigil events">
        <div className="event-row event-labels" role="row">
          <span>Time</span>
          <span>Type</span>
          <span>Source</span>
          <span>Event</span>
          <span>Patient</span>
        </div>

        <div className="event-rows">
          {displayRows.length === 0 ? (
            <div className="event-empty">
              <span className="empty-line" aria-hidden="true" />
              <p>Event stream ready. New sensor and agent activity will appear here.</p>
            </div>
          ) : (
            displayRows.map((row) => (
              <div key={row.id} className={`event-row ${tone(row.type)}`} role="row">
                <time>{eventTime(row.created_at)}</time>
                <span className="event-type">{cleanText(TYPE_LABEL[row.type] ?? row.type)}</span>
                <span className="event-source">{cleanText(row.source) || "System"}</span>
                <span className="event-summary">{cleanText(row.summary) || cleanText(row.type)}</span>
                <span className="event-patient">{cleanText(row.patient) || "None"}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
