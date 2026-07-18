export type VigilRole =
  | "charge_nurse"
  | "attending"
  | "triage_nurse"
  | "front_desk"
  | "security"
  | "compliance";

export type AlertView = {
  alert_id: string;
  patient_id: string;
  severity: "soft" | "hard";
  state: "checkin" | "page_pending" | "acknowledged" | "resolved";
  title: string;
  evidence?: string[];
  prior_esi: number;
  current_esi: number;
  acknowledged_by?: string | null;
};

export type QueuePatient = {
  patient_id?: string;
  patient_ref?: string;
  name?: string;
  age?: number;
  gender?: string;
  seat?: string;
  visit?: string;
  initial_esi?: number;
  current_esi?: number;
  wait_minutes?: number;
  priority?: number;
  status?: string;
  latest_signal?: string;
  baseline_deviation?: number;
  risk_factors?: string[];
  reassessment_due?: boolean;
  preferred_language?: string;
  accessibility_mode?: string;
  flagged?: boolean;
  medical_assist_needed?: boolean;
  alert_type?: string | null;
  alert?: AlertView | null;
  chart?: {
    conditions: string[];
    medications: string[];
    vitals: Record<string, { value: number; unit: string; at: string }>;
  };
  redacted?: boolean;
};

export type AuditView = {
  index: number;
  audit_id: string;
  ts: number;
  actor: string;
  role: string;
  action: string;
  resource: string;
  outcome: string;
  reason?: string | null;
  hash: string;
};

export type VigilSession = {
  role: VigilRole;
  scopes: string[];
  queue: QueuePatient[];
  alert_budget: { used: number; limit: number };
  demo: { step: number; total_steps: number };
  audit_verified: { valid: boolean; blocks: number; head?: string };
  audit?: AuditView[];
  source?: "backend" | "local";
};

export const ROLES: Array<{ value: VigilRole; label: string }> = [
  { value: "charge_nurse", label: "Charge nurse" },
  { value: "attending", label: "Attending" },
  { value: "triage_nurse", label: "Triage nurse" },
  { value: "front_desk", label: "Front desk" },
  { value: "security", label: "Security" },
  { value: "compliance", label: "Compliance" },
];

const basePatients: QueuePatient[] = [
  {
    patient_id: "demo-vega",
    name: "Maria Vega",
    age: 67,
    gender: "female",
    seat: "A3",
    visit: "Chest pressure and shortness of breath",
    initial_esi: 3,
    current_esi: 3,
    wait_minutes: 128,
    priority: 72,
    status: "monitoring",
    latest_signal: "Stable baseline",
    baseline_deviation: 0.08,
    risk_factors: ["anticoagulated", "cardiac history", "low oxygen saturation"],
    reassessment_due: true,
    preferred_language: "Spanish",
    accessibility_mode: "voice and text",
    alert: null,
    chart: {
      conditions: ["Coronary artery disease", "Hypertension"],
      medications: ["Apixaban 5 mg", "Metoprolol 50 mg"],
      vitals: {
        heart_rate: { value: 108, unit: "/min", at: "2026-07-18T18:10:00Z" },
        spo2: { value: 91, unit: "%", at: "2026-07-18T18:10:00Z" },
      },
    },
  },
  {
    patient_id: "demo-idris",
    name: "Idris Cole",
    age: 54,
    gender: "male",
    seat: "B1",
    visit: "Abdominal pain and dizziness",
    initial_esi: 4,
    current_esi: 4,
    wait_minutes: 82,
    priority: 52,
    status: "monitoring",
    latest_signal: "Stable baseline",
    baseline_deviation: 0.06,
    risk_factors: ["diabetes", "high comorbidity"],
    reassessment_due: true,
    preferred_language: "English",
    accessibility_mode: "voice and text",
    alert: null,
    chart: {
      conditions: ["Type 2 diabetes mellitus", "Chronic kidney disease"],
      medications: ["Insulin glargine", "Lisinopril 10 mg"],
      vitals: {
        heart_rate: { value: 96, unit: "/min", at: "2026-07-18T18:25:00Z" },
        systolic_bp: { value: 102, unit: "mmHg", at: "2026-07-18T18:25:00Z" },
      },
    },
  },
  {
    patient_id: "demo-park",
    name: "Jordan Park",
    age: 29,
    gender: "nonbinary",
    seat: "C2",
    visit: "Ankle injury after a fall",
    initial_esi: 4,
    current_esi: 4,
    wait_minutes: 37,
    priority: 37,
    status: "monitoring",
    latest_signal: "Stable baseline",
    baseline_deviation: 0.04,
    risk_factors: [],
    reassessment_due: false,
    preferred_language: "English",
    accessibility_mode: "text preferred",
    alert: null,
    chart: {
      conditions: [],
      medications: [],
      vitals: {
        heart_rate: { value: 78, unit: "/min", at: "2026-07-18T18:42:00Z" },
      },
    },
  },
];

const scopeMap: Record<VigilRole, string[]> = {
  charge_nurse: ["chart:read", "escalate:ack", "esi:override", "queue:read", "reason:read", "video:view"],
  attending: ["chart:read", "esi:override", "queue:read", "reason:read"],
  triage_nurse: ["chart:read", "queue:read"],
  front_desk: ["queue:read:limited"],
  security: ["alert:read:nonclinical"],
  compliance: ["audit:read"],
};

function clinicalPatients(step: number): QueuePatient[] {
  const patients = structuredClone(basePatients);
  const vega = patients[0];
  const idris = patients[1];
  const park = patients[2];

  if (step >= 1) {
    vega.status = "checkin";
    vega.latest_signal = "Posture declining for 74 seconds";
    vega.baseline_deviation = 0.46;
    vega.priority = 88;
    vega.alert = {
      alert_id: "alert-vega",
      patient_id: "demo-vega",
      severity: "soft",
      state: "checkin",
      title: "Posture change detected",
      evidence: ["posture declined from baseline"],
      prior_esi: 3,
      current_esi: 3,
    };
  }
  if (step >= 2) {
    vega.current_esi = 2;
    vega.status = "page_pending";
    vega.latest_signal = "Visual and respiratory signals corroborated";
    vega.baseline_deviation = 0.86;
    vega.priority = 132;
    vega.alert = {
      alert_id: "alert-vega",
      patient_id: "demo-vega",
      severity: "hard",
      state: "page_pending",
      title: "Corroborated visual and audio deterioration",
      evidence: [
        "posture declined from baseline",
        "possible labored breathing",
        "cardiac history",
        "low charted oxygen saturation",
      ],
      prior_esi: 3,
      current_esi: 2,
    };
  }
  if (step >= 3) {
    park.status = "checkin";
    park.latest_signal = "Ambiguous movement above baseline";
    park.baseline_deviation = 0.29;
    park.priority = 61;
    park.alert = {
      alert_id: "alert-park",
      patient_id: "demo-park",
      severity: "soft",
      state: "checkin",
      title: "Movement above baseline",
      evidence: ["single low-confidence visual signal"],
      prior_esi: 4,
      current_esi: 4,
    };
  }
  if (step >= 4) {
    idris.current_esi = 2;
    idris.status = "page_pending";
    idris.latest_signal = "Companion requested urgent help";
    idris.baseline_deviation = 0.81;
    idris.priority = 118;
    idris.alert = {
      alert_id: "alert-idris",
      patient_id: "demo-idris",
      severity: "hard",
      state: "page_pending",
      title: "Companion requested urgent help",
      evidence: ["companion moved rapidly toward staff", "high comorbidity"],
      prior_esi: 4,
      current_esi: 2,
    };
  }
  return patients.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
}

function redactedPatients(role: VigilRole, patients: QueuePatient[]): QueuePatient[] {
  if (role === "front_desk") {
    return patients.map((patient) => ({
      patient_id: patient.patient_id,
      name: patient.name,
      seat: patient.seat,
      wait_minutes: patient.wait_minutes,
      flagged: ["checkin", "page_pending"].includes(patient.status ?? ""),
      redacted: true,
    }));
  }
  if (role === "security") {
    return patients.map((patient) => ({
      patient_ref: `pt_${patient.patient_id?.replace("demo-", "").slice(0, 6)}`,
      seat: patient.seat,
      medical_assist_needed: ["checkin", "page_pending"].includes(patient.status ?? ""),
      alert_type: ["checkin", "page_pending"].includes(patient.status ?? "") ? "medical assist" : null,
      redacted: true,
    }));
  }
  if (role === "compliance") {
    return patients.map((patient) => ({
      patient_ref: `pt_${patient.patient_id?.replace("demo-", "").slice(0, 6)}`,
      wait_minutes: patient.wait_minutes,
      flagged: ["checkin", "page_pending"].includes(patient.status ?? ""),
      redacted: true,
    }));
  }
  if (role === "triage_nurse") {
    return patients.map((patient) => {
      const copy = structuredClone(patient);
      if (copy.alert) delete copy.alert.evidence;
      return copy;
    });
  }
  return patients;
}

function auditBlocks(step: number): AuditView[] {
  const demoEpoch = 1784397600;
  const blocks: AuditView[] = [
    {
      index: 0,
      audit_id: "audit-0001",
      ts: demoEpoch,
      actor: "demo-runner",
      role: "charge_nurse",
      action: "demo_reset",
      resource: "waiting_room",
      outcome: "success",
      hash: "9f31c07d2a81",
    },
  ];
  if (step >= 1) {
    blocks.push({
      index: 1,
      audit_id: "audit-0002",
      ts: demoEpoch + 6,
      actor: "vigil-tier-0",
      role: "system",
      action: "patient_checkin",
      resource: "patient:pt_vega",
      outcome: "started",
      hash: "b845e209f186",
    });
  }
  if (step >= 2) {
    blocks.push({
      index: 2,
      audit_id: "audit-0003",
      ts: demoEpoch + 12,
      actor: "vigil-tier-0",
      role: "system",
      action: "retriage_decision",
      resource: "patient:pt_vega",
      outcome: "escalated",
      hash: "e33dc74ab921",
    });
  }
  if (step >= 3) {
    blocks.push({
      index: 3,
      audit_id: "audit-0004",
      ts: demoEpoch + 15,
      actor: "demo-front_desk",
      role: "front_desk",
      action: "chart_read",
      resource: "patient:pt_park",
      outcome: "denied",
      hash: "a102bc736fa0",
    });
  }
  return blocks;
}

export function localSession(role: VigilRole, step = 0): VigilSession {
  const patients = clinicalPatients(step);
  const audit = auditBlocks(step);
  return {
    role,
    scopes: scopeMap[role],
    queue: redactedPatients(role, patients),
    alert_budget: { used: patients.filter((patient) => patient.alert).length, limit: 8 },
    demo: { step, total_steps: 5 },
    audit_verified: {
      valid: true,
      blocks: audit.length,
      head: audit.at(-1)?.hash,
    },
    audit: role === "compliance" || role === "charge_nurse" ? audit : undefined,
    source: "local",
  };
}

export async function fetchVigilSession(backend: string, role: VigilRole): Promise<VigilSession> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 2200);
  try {
    const response = await fetch(`${backend}/api/v1/session`, {
      headers: { "X-Vigil-Role": role },
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Session request failed with ${response.status}`);
    return { ...(await response.json()), source: "backend" } as VigilSession;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function postVigilCommand(
  backend: string,
  path: string,
  role: VigilRole,
  body: Record<string, unknown> = {},
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 1600);
  try {
    return await fetch(`${backend}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Vigil-Role": role },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeout);
  }
}
