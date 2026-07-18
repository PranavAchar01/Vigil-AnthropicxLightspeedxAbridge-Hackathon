type PipelineProps = {
  capabilities?: Record<string, boolean>;
  source?: "backend" | "local";
  compact?: boolean;
};

const stages = [
  {
    index: "01",
    agent: "Observer",
    task: "Pose and audio signals",
    capability: "multi_patient",
  },
  {
    index: "02",
    agent: "Fusion",
    task: "Patient-scoped corroboration",
    capability: "multi_patient",
  },
  {
    index: "03",
    agent: "Re-triage",
    task: "Tier 0 rules and Claude",
    capability: "reasoning",
  },
  {
    index: "04",
    agent: "Escalation",
    task: "Check-in and nurse routing",
    capability: "nurse_call",
  },
  {
    index: "05",
    agent: "Documentation",
    task: "SOAP, FHIR, and provenance",
    capability: "audit_chain",
  },
];

export default function AgentPipeline({ capabilities = {}, source = "local", compact = false }: PipelineProps) {
  return (
    <section className={`agent-pipeline surface ${compact ? "compact" : ""}`} aria-label="Vigil agent pipeline">
      <div className="agent-pipeline-heading">
        <div>
          <span className="eyebrow">One coordinated system</span>
          <h2>Live agent pipeline</h2>
        </div>
        <span className={`source-pill ${source === "backend" ? "connected" : ""}`}>
          {source === "backend" ? "Backend connected" : "Stage-safe preview"}
        </span>
      </div>
      <div className="agent-pipeline-grid">
        {stages.map((stage, index) => {
          const configured = capabilities[stage.capability];
          const coreActive = ["multi_patient", "audit_chain"].includes(stage.capability);
          return (
            <article className="agent-stage" key={stage.agent}>
              <span className="agent-index">{stage.index}</span>
              <div>
                <strong>{stage.agent}</strong>
                <p>{stage.task}</p>
              </div>
              <small className={configured || coreActive ? "ready" : "optional"}>
                {configured || coreActive ? "Ready" : "Optional service"}
              </small>
              {index < stages.length - 1 ? <i aria-hidden="true" /> : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
