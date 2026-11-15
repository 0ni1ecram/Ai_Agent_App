#!/usr/bin/env python3
"""KURTA workflow data contracts — every step, typed.

One source of truth for the whole request lifecycle. Run this file to
validate every sample against its model and emit kurta_workflow_samples.json,
which the build blueprint (blueprint/index.html) quotes verbatim.

Pipeline:
  S1 goal-intake -> S2 goal-parse GoalSpec -> S3 plan-compile CompiledPlan
  -> S4 plan-validate (DAG rules, in-code) -> S5 execute PlanStep via
  ToolCallRequest/ToolResult -> S6 ingest unstructured IngestedItem ->
  S7 score anomaly AnomalyScore -> S8 policy check PolicyInput /
  PolicyDecision (OPA) -> S9 self-correct SelfCorrectionEvent ->
  S10 HITL ApprovalRequest / ApprovalDecision -> S11 mutate MutationReceipt
  -> S12 append TraceRow -> S13 aggregate RoiAggRow

Money is always integer minor units; time is always aware UTC; identifiers
are opaque strings. No free-form dicts cross a step boundary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, AfterValidator, model_validator

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# shared primitives
# --------------------------------------------------------------------------
class Base(BaseModel):
    """Strict config for every contract: unknown fields are errors, not typos."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_bytes="base64",
        use_enum_values=True,
    )


Id = Annotated[str, Field(min_length=8, max_length=40, pattern=r"^[a-z0-9_]+$")]
# ULID-style, e.g. "run_01j8y6c2qe5t7wxz3vbnhm9k4p"


class Money(Base):
    """Integer minor units only. float money is banned across the platform."""

    amount_minor: int = Field(description="signed cents / minor units")
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]

    @property
    def major(self) -> Decimal:
        return Decimal(self.amount_minor) / 100


class ActorRef(Base):
    """Human or workload identity; agents carry the delegator they inherit from."""

    kind: Literal["human", "agent", "service"]
    id: Id
    roles: list[str] = Field(default_factory=list)
    delegator: Optional[Id] = None

    @model_validator(mode="after")
    def agent_needs_delegator(self) -> "ActorRef":
        if self.kind == "agent" and not self.delegator:
            raise ValueError("agent actors must name a delegating human identity")
        return self


class SpanRef(Base):
    """A citation that cannot drift: text plus the exact version it came from."""

    doc_id: Id
    doc_version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(description="e.g. '§4.2' or 'p.4 lines 12-18'")
    quoted_text: str


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("naive datetimes are rejected; UTC aware only")
    return v.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]


def sha(seed: str) -> str:
    """Deterministic 64-hex digest for samples (stands in for real content hashes)."""
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


# --------------------------------------------------------------------------
# S1-S2: intake & parse
# --------------------------------------------------------------------------
class ContextRefKind(str, Enum):
    EMAIL = "email"
    SLACK_THREAD = "slack_thread"
    TICKET = "ticket"
    DOCUMENT = "document"
    CONVERSATION = "conversation"


class ContextRef(Base):
    kind: ContextRefKind
    locator: str = Field(description="Message-ID, slack permalink, JIRA key, doc id")
    note: Optional[str] = None


class GoalIntake(Base):
    """S1 · what the human actually said. Raw text in; nothing interpreted yet."""

    run_id: Id
    tenant_id: Id
    submitted_by: ActorRef
    text: str = Field(min_length=8, max_length=8000)
    contexts: list[ContextRef] = Field(default_factory=list)
    created_at: UtcDatetime


class AmbiguityFlag(Base):
    field: str
    question: str
    blocking: bool = True


class GoalDeliverable(str, Enum):
    FINDINGS_REPORT = "findings_report"
    PURCHASE_ORDER_DRAFT = "purchase_order_draft"
    CONTRACT_REVIEW = "contract_review"
    APPROVAL_PACKET = "approval_packet"


class GoalSpec(Base):
    """S2 · parsed intent. The LLM emits this typed object or the parse fails -
    there is no prose between S2 and S3."""

    schema_version: Annotated[str, Field(pattern=r"^goalspec/v\d+$")] = "goalspec/v1"
    run_id: Id
    objective: str = Field(max_length=500)
    deliverable: GoalDeliverable
    deadline: Optional[UtcDatetime] = None
    budget_cap: Optional[Money] = None
    must_include_human: list[ActorRef] = Field(default_factory=list)
    in_scope_systems: list[Annotated[str, Field(pattern=r"^[a-z0-9_.-]+$")]] = Field(
        default_factory=list
    )
    out_of_scope: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    parsed_confidence: Annotated[float, Field(ge=0, le=1)]
    ambiguity_flags: list[AmbiguityFlag] = Field(default_factory=list)

    @model_validator(mode="after")
    def blocking_ambiguity_hands_control_back(self) -> "GoalSpec":
        blockers = [f for f in self.ambiguity_flags if f.blocking]
        if blockers and self.parsed_confidence < 0.85:
            raise ValueError("blocking ambiguity requires a clarification round, not a plan")
        return self



# --------------------------------------------------------------------------
# S3-S4: plan compile & validate
# --------------------------------------------------------------------------
class ToolOp(str, Enum):
    SNOWFLAKE_QUERY = "snowflake.query"
    NETSUITE_RECORD_GET = "netsuite.record.get"
    NETSUITE_PO_CREATE = "netsuite.po.create"
    JIRA_ISSUE_SEARCH = "jira.issue.search"
    SLACK_POST_DRAFT = "slack.post.draft"
    DOC_PARSE_INGEST = "docparse.ingest"
    POLICY_EVAL = "opa.evaluate"
    COMPUTE_FORECAST = "compute.forecast"
    COMPUTE_VENDOR_SCORE = "compute.vendor_score"
    EMAIL_EXTERNAL_SEND = "email.external.send"  # always policy-gated
    DOCUSIGN_ENVELOPE_SEND = "docsign.envelope_send"
    SLACK_POST_PUBLISH = "slack.post.publish"


class IOContract(Base):
    """Every step declares exact upstream output field names it consumes."""

    consumes: list[str] = Field(default_factory=list, description="'step_id.field' paths")
    produces: list[str] = Field(default_factory=list)


class PlanStep(Base):
    step_id: int = Field(ge=1)
    op: ToolOp
    args_schema_ref: str = Field(description="registry pointer, not inline args")
    io: IOContract
    depends_on: list[int] = Field(default_factory=list)
    mutates_state: bool = False
    spend_minor: Optional[Money] = None
    gate: Optional[str] = Field(default=None, description="policy id that must allow first")
    max_retries: Annotated[int, Field(ge=0, le=5)] = 3
    est_llm_cost_usd_micro: Annotated[int, Field(ge=0)] = 0
    playbook_id: Optional[Id] = None

    @model_validator(mode="after")
    def side_effects_are_always_gated(self) -> "PlanStep":
        if (self.mutates_state or self.op == ToolOp.EMAIL_EXTERNAL_SEND) and not self.gate:
            raise ValueError("a step that mutates state or sends externally must name its policy gate")
        return self


class CompiledPlan(Base):
    """S3 · planner output. S4 validates the rules below, in code, before run."""

    schema_version: Annotated[str, Field(pattern=r"^plan/v\d+$")] = "plan/v1"
    run_id: Id
    steps: list[PlanStep] = Field(min_length=1)
    goal_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_dag(self) -> "CompiledPlan":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step_id in plan")
        known = set(ids)
        for s in self.steps:
            for d in s.depends_on:
                if d not in known:
                    raise ValueError(f"step {s.step_id} depends on unknown {d}")
        # Kahn's algorithm: plan must be acyclic
        indeg = {i: 0 for i in known}
        adj: dict[int, list[int]] = {i: [] for i in known}
        for s in self.steps:
            for d in s.depends_on:
                adj[d].append(s.step_id)
                indeg[s.step_id] += 1
        queue = [i for i, v in indeg.items() if v == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen != len(known):
            raise ValueError("plan DAG has a cycle")
        # mutation/external-send gating is enforced per PlanStep at construction —
        # an ungated mutating step can never reach a CompiledPlan in the first place.
        return self


# --------------------------------------------------------------------------
# S5-S6: tool execution & unstructured ingestion
# --------------------------------------------------------------------------
class ColumnDType(Base):
    """Arrow-style physical types. DataFrames cross steps as declared schemas."""

    name: str
    arrow_type: Literal[
        "int64", "float64", "decimal128(18,4)", "string", "bool", "timestamp[us, tz=UTC]"
    ]
    nullable: bool = False


class RowSet(Base):
    columns_dtypes: list[ColumnDType]
    rows: list[list[Any]]

    @model_validator(mode="after")
    def rows_match_schema(self) -> "RowSet":
        width = len(self.columns_dtypes)
        for i, r in enumerate(self.rows):
            if len(r) != width:
                raise ValueError(f"row {i}: {len(r)} cells vs {width} declared dtypes")
        return self


class ErrorKind(str, Enum):
    AUTH = "AUTH"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    RATE_LIMIT = "RATE_LIMIT"
    SEMANTIC = "SEMANTIC"
    TIMEOUT = "TIMEOUT"


class ToolError(Base):
    kind: ErrorKind
    detail: str
    retry_after_ms: Optional[int] = None
    observed_schema_version: Optional[str] = None
    expected_schema_version: Optional[str] = None


ToolArgs = Annotated[dict, Field(description="must match args_schema_ref in registry")]


class ToolCallRequest(Base):
    """S5 · the only way an agent touches an external system. Never raw HTTP."""

    call_id: Id
    run_id: Id
    step_id: int
    tool: ToolOp
    args: ToolArgs
    actor: ActorRef
    idempotency_key: Id
    max_rows: Annotated[int, Field(ge=1, le=2_000_000)] = 100_000
    timeout_ms: Annotated[int, Field(ge=100, le=120_000)] = 30_000


class ToolResult(Base):
    call_id: Id
    ok: bool
    data: Optional[RowSet] = None
    scalar: Optional[Any] = None
    error: Optional[ToolError] = None
    observed_schema_version: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict, description="credits/api calls")

    @model_validator(mode="after")
    def shape_follows_ok(self) -> "ToolResult":
        if self.ok and (self.data is None and self.scalar is None):
            raise ValueError("ok result must carry data or scalar")
        if not self.ok and self.error is None:
            raise ValueError("failed result must carry a typed error")
        return self


class ClauseEntity(Base):
    """S6 · one extracted, source-pinned fact from a document/email/audio."""

    field: str
    value: Any
    dtype_name: Literal["decimal", "int64", "string", "date", "bool"]
    span: SpanRef
    extractor: Annotated[str, Field(description="model+version, e.g. qwen-doc-extract/v3.2")]
    confidence: Annotated[float, Field(ge=0.7, le=1, description="below 0.7 goes to the human review queue, never into the pipeline")]


class IngestedItem(Base):
    source_ref: ContextRef
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    doc_version_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    acl_vector: list[str] = Field(description="copy of source-system ACL scopes at ingest")
    entities: list[ClauseEntity] = Field(min_length=1)
    redactions_applied: list[Literal["PII", "PAYROLL", "LEGAL"]] = Field(default_factory=list)
    parsed_at: UtcDatetime


# --------------------------------------------------------------------------
# S7-S9: anomaly scoring, policy decision, self-correction
# --------------------------------------------------------------------------
class FeatureVector(Base):
    """Typed tensor the Isolation-Forest/GBM screen consumes. Fixed order."""

    feature_names: list[str]
    values_f64: list[float]
    row_key: str
    split_candidate_i8: Annotated[int, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def lens(self) -> "FeatureVector":
        if len(self.feature_names) != len(self.values_f64):
            raise ValueError("feature vector shape mismatch")
        return self


class AnomalyVerdict(str, Enum):
    CLEAN = "clean"
    FLAG = "flag"
    ESCALATE = "escalate"


class AnomalyScore(Base):
    model_id: str
    score: Annotated[float, Field(ge=0, le=1)]
    threshold: Annotated[float, Field(ge=0, le=1)]
    verdict: AnomalyVerdict
    policy_hit: Optional[SpanRef] = None
    human_explanation: str = Field(max_length=400)
    abstain: bool = False

    @model_validator(mode="after")
    def low_conf_abstains(self) -> "AnomalyScore":
        if abs(self.score - self.threshold) < 0.02:
            self.abstain = True
            self.verdict = AnomalyVerdict.ESCALATE
        if self.verdict in (AnomalyVerdict.FLAG, AnomalyVerdict.ESCALATE) and self.policy_hit is None:
            raise ValueError("a flag without a cited policy clause is noise, not an audit finding")
        return self


class PolicyAction(Base):
    tool: ToolOp
    spend: Optional[Money] = None
    external_delivery: bool = False
    touches: list[str] = Field(default_factory=list, description="data classes: payroll, legal, PII")


class PolicyInput(Base):
    """S8 · OPA input document. Deterministic: same input, same decision, provable."""

    subject: dict  # ActorRef dumped + agent cumulative spend this period
    action: PolicyAction
    resource: dict = Field(default_factory=dict, description="acl_vector of touched data")


class PolicyDecision(Base):
    schema_version: Annotated[str, Field(pattern=r"^decision/v\d+$")] = "decision/v1"
    policy_bundle_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    allow: bool
    gate: Literal["auto", "hco", "HITL_REQUIRED", "DENY"]
    obligations: list[str] = Field(default_factory=list)
    reason: str = Field(max_length=600)
    matched_rule: str


CorrectStrategyKind = Literal[
    "FIELD_ALIAS_REMAP",
    "ALTERNATE_SOURCE",
    "RETRY_BACKOFF",
    "REPLAN_SUBGRAPH",
    "ESCALATE_HUMAN",
]


class SelfCorrectionEvent(Base):
    """S9 · schema drift and failures become auditable events, not crashes."""

    run_id: Id
    step_id: int
    trigger: ToolError
    strategy: CorrectStrategyKind
    mapping_detail: Optional[dict] = None
    rows_revalidated_i64: Optional[int] = None
    outcome: Literal["RECOVERED", "RETRYING", "ESCALATED"]
    extra_latency_ms: int = Field(ge=0)


# --------------------------------------------------------------------------
# S10-S13: HITL, mutation, trace, ROI
# --------------------------------------------------------------------------
class BlastRadiusItem(Base):
    tool: ToolOp
    effect: str
    reversible: Literal["yes", "no", "with_counter_txn"]


class ApprovalRequest(Base):
    """S10 · the human gate. Carries everything needed to decide, nothing more."""

    approval_id: Id
    run_id: Id
    step_id: int
    title: str
    recommendation: Literal["approve", "modify_then_approve", "reject"]
    recommendation_basis: str
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="pre-flight checks + cites (SpanRef[])"
    )
    blast_radius: list[BlastRadiusItem] = Field(min_length=1)
    spend: Optional[Money] = None
    policy_decision_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    required_signers: list[Id]
    min_signatures: Annotated[int, Field(ge=1)] = 1
    created_at: UtcDatetime
    expires_at: UtcDatetime

    @model_validator(mode="after")
    def window_positive(self) -> "ApprovalRequest":
        if self.expires_at <= self.created_at:
            raise ValueError("approval expiry must be after creation")
        return self


class ApprovalDecision(Base):
    approval_id: Id
    decision: Literal["APPROVE", "REJECT_WITH_REASON", "CONDITION_ADDED"]
    conditions_text: Optional[str] = None
    signer: ActorRef
    signature_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    signed_at: UtcDatetime
    elapsed_since_request_ms: int = Field(ge=0, description="feeds approval latency SLO")

    @model_validator(mode="after")
    def reject_has_reason(self) -> "ApprovalDecision":
        if self.decision == "REJECT_WITH_REASON" and not self.conditions_text:
            raise ValueError("rejections without a reason never teach the agent")
        return self


class MutationReceipt(Base):
    """S11 · proof the gated write happened, exactly once, as previewed."""

    receipt_id: Id
    approval_decision_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tool: ToolOp
    external_id: str = Field(description="e.g. PO-9407, DocuSign envelope id")
    state: Literal["posted", "queued_external", "compensated"]
    idempotency_key: Id
    observed_schema_version: Optional[str] = None
    mutated_at: UtcDatetime


class TraceRow(Base):
    """S12 · one hash-chained, append-only line of machine-readable reasoning."""

    seq: Annotated[int, Field(ge=0)]
    prev_hash_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    ts: UtcDatetime
    run_id: Id
    step_id: Optional[int] = None
    actor: ActorRef
    op: Literal[
        "goal.intake", "goal.parse", "plan.compile", "plan.validate", "tool.request",
        "tool.result", "ingest.parse", "anomaly.score", "policy.decide", "selfcorrect",
        "hitl.request", "hitl.decide", "mutate", "trace.append", "roi.aggregate",
    ]
    refs: dict[str, str] = Field(default_factory=dict, description="ids/hashes of payload objects")
    llm_ctx: Optional[dict[str, Any]] = Field(
        default=None, description="model_version, prompt_sha256, seed, temperature"
    )
    cost_usd_micro: Annotated[int, Field(ge=0)] = 0


class RoiAggRow(Base):
    """S13 · the K09 numbers. Everything below derives from TraceRows only."""

    agent_id: Id
    period: Literal["weekly", "monthly", "quarterly"]
    started_at: UtcDatetime
    hours_offloaded_dec: Annotated[float, Field(ge=0)]
    runs_completed_i64: Annotated[int, Field(ge=0)]
    approval_latency_p50_s: Annotated[float, Field(ge=0)]
    hitl_acceptance_rate: Annotated[float, Field(ge=0, le=1)]
    selfcorrect_recovered_pct: Annotated[float, Field(ge=0, le=100)]
    gated_value_usd_minor_i64: int
    run_cost_usd_micro_i64: Annotated[int, Field(ge=0)]
    policy_misses_i64: Annotated[int, Field(ge=0)] = 0

    @property
    def value_to_cost_ratio(self) -> float:
        return (
            (self.gated_value_usd_minor_i64 / 100)
            / max(self.run_cost_usd_micro_i64 / 1_000_000, 1e-9)
        )


# --------------------------------------------------------------------------
# samples + runner
# --------------------------------------------------------------------------
RUN = "run_01j8y6c2qe5t7wxz3vbnhm9k4p"

HUMAN = ActorRef(kind="human", id="usr_marcelino", roles=["ops_lead", "approver_ops"])
AGENT_AUDIT = ActorRef(kind="agent", id="agent_auditor_2", roles=["finance_agent"], delegator="usr_marcelino")

S1_INPUT = GoalIntake(
    run_id=RUN,
    tenant_id="acme_group",
    submitted_by=HUMAN,
    text="Find out why Q3 shipping costs spiked and bring me a plan to cut them — "
    "talk to procurement before proposing anything we can't fund.",
    contexts=[
        ContextRef(kind="email", locator="msgid:<20261009-ops-apac@mailer>", note="manager thread"),
        ContextRef(kind="slack_thread", locator="https://acme.slack.com/archives/C0APAC/p17285472000"),
    ],
    created_at="2026-11-14T13:30:02Z",
)

S2_OUTPUT = GoalSpec(
    run_id=RUN,
    objective="Quantify the drivers of the Q3 freight cost variance and propose a fundable reduction plan",
    deliverable="findings_report",
    deadline="2026-11-17T17:00:00Z",
    budget_cap=Money(amount_minor=500_000_00, currency="USD"),
    must_include_human=[ActorRef(kind="human", id="usr_proc_lead", roles=["procurement"])],
    in_scope_systems=["snowflake", "sap", "jira", "slack", "docstore"],
    out_of_scope=["payroll", "hr"],
    acceptance_criteria=[
        "variance attributed to contract clause vs demand with numbers",
        "every claim cites a policy/contract span with doc version hash",
        "reduction options priced against Q4 budget lines",
    ],
    parsed_confidence=0.91,
    ambiguity_flags=[
        AmbiguityFlag(field="deadline", question="end of business Monday?", blocking=False)
    ],
)

S3_OUTPUT = CompiledPlan(
    run_id=RUN,
    goal_sha256=sha("goal:q3-freight-2026"),
    steps=[
        PlanStep(
            step_id=1, op="snowflake.query", args_schema_ref="registry/snowflake.query@v4",
            io=IOContract(produces=["lane", "freight_spend_usd", "period"]),
            spend_minor=None, est_llm_cost_usd_micro=4_200, playbook_id="pbq_freight_variance_v6",
        ),
        PlanStep(
            step_id=2, op="docparse.ingest", args_schema_ref="registry/docparse.ingest@v2",
            io=IOContract(consumes=["step1.lane"], produces=["carrier_clauses"]),
        ),
        PlanStep(
            step_id=3, op="compute.forecast", args_schema_ref="registry/compute.forecast@v3",
            io=IOContract(consumes=["step1.freight_spend_usd", "step2.carrier_clauses"],
                          produces=["attribution_table", "savings_plan"]),
            depends_on=[1, 2],
        ),
        PlanStep(
            step_id=4, op="netsuite.po.create", args_schema_ref="registry/netsuite.po.create@v3",
            io=IOContract(consumes=["step3.savings_plan.renewal_sku"]),
            depends_on=[3], mutates_state=True,
            spend_minor=Money(amount_minor=14_200_000, currency="USD"),
            gate="finance/spend_50k_co_sign", max_retries=0,
        ),
        PlanStep(
            step_id=5, op="slack.post.draft", args_schema_ref="registry/slack.post.draft@v2",
            io=IOContract(consumes=["step4.po_id", "step3.attribution_table"]),
            depends_on=[4],
        ),
    ],
)


S5_INPUT = ToolCallRequest(
    call_id="call_01j8y6d9m1k8",
    run_id=RUN,
    step_id=1,
    tool="snowflake.query",
    args={
        "role": "cur_finance_ro",
        "sql_template_id": "freight_variance_q3_v2",
        "params": {"period": "2026Q3", "base": "2026Q2", "lanes": "APAC"},
        "row_limit": 100_000,
    },
    actor=AGENT_AUDIT,
    idempotency_key="idem_s1_q3var",
    timeout_ms=30_000,
)

S5_OUTPUT = ToolResult(
    call_id="call_01j8y6d9m1k8",
    ok=True,
    data=RowSet(
        columns_dtypes=[
            ColumnDType(name="lane", arrow_type="string"),
            ColumnDType(name="period", arrow_type="string"),
            ColumnDType(name="freight_spend_usd", arrow_type="decimal128(18,4)", nullable=True),
            ColumnDType(name="is_surcharge_bearing", arrow_type="bool"),
            ColumnDType(name="captured_at", arrow_type="timestamp[us, tz=UTC]"),
        ],
        rows=[
            ["CN-NL->DE", "2026Q3", 1_284_509.24, True, "2026-11-14T14:02:11Z"],
            ["CN-NL->DE", "2026Q2", 1_148_002.10, False, "2026-11-14T14:02:11Z"],
            ["SG->FRA", "2026Q3", 452_118.77, True, "2026-11-14T14:02:11Z"],
        ],
    ),
    observed_schema_version="warehouse_fact_freight.v11",
)

S6_OUTPUT = IngestedItem(
    source_ref=ContextRef(kind="document", locator="docstore://carrier_contracts/gilead_2026.pdf"),
    content_sha256=sha("doc:gilead_2026.pdf"),
    doc_version_sha256=sha("ver:gilead_2026/v3"),
    acl_vector=["scope:finance", "scope:legal_read", "deny:hr", "deny:payroll"],
    entities=[
        ClauseEntity(
            field="unit_price_cap", value="42.00", dtype_name="decimal",
            span=SpanRef(doc_id="gilead_2026_contract", doc_version_sha256=sha("ver:gilead_2026/v3"),
                         locator="p.4 §3.1 lines 12-18",
                         quoted_text="unit price fixed at $42, adjustments require CFO countersignature"),
            extractor="qwen-doc-extract/v3.2", confidence=0.99,
        ),
        ClauseEntity(
            field="surcharge_clause_present", value=True, dtype_name="bool",
            span=SpanRef(doc_id="gilead_2026_contract", doc_version_sha256=sha("ver:gilead_2026/v3"),
                         locator="p.9 §4.2",
                         quoted_text="fuel surcharge applies at published index rate to base lanes"),
            extractor="qwen-doc-extract/v3.2", confidence=0.97,
        ),
    ],
    redactions_applied=["PII"],
    parsed_at="2026-11-14T14:05:40Z",
)

S7_INPUT = FeatureVector(
    feature_names=["amount_delta_pct", "qty_delta_pct", "timing_bucket_z", "rate_vs_clause_cap", "split_floor_distance"],
    values_f64=[14.3, 1.2, 0.8, 12.5, 0.03],
    row_key="INV-7719",
    split_candidate_i8=0,
)

S7_OUTPUT = AnomalyScore(
    model_id="iforest_invoice_screen@2026-09",
    score=0.98, threshold=0.80, verdict="flag",
    policy_hit=SpanRef(doc_id="gilead_2026_contract", doc_version_sha256=sha("ver:gilead_2026/v3"),
                       locator="p.4 §3.1", quoted_text="unit price fixed at $42"),
    human_explanation="Unit price $48.00 exceeds contract cap $42.00 by 14% on 5,200 units; "
    "invoice is otherwise well-formed — the finding is contract, not fraud.",
)

S8_INPUT = PolicyInput(
    subject={
        "agent_id": "agent_auditor_2", "roles": ["finance_agent"], "delegator": "usr_marcelino",
        "spend_approved_cum_minor": 0, "clearance": ["scope:finance"],
    },
    action=PolicyAction(
        tool="netsuite.po.create",
        spend=Money(amount_minor=14_200_000, currency="USD"),
        external_delivery=True, touches=["vendor_terms"],
    ),
    resource={"acl_vector": ["scope:finance", "scope:vendor"], "system": "netsuite"},
)

S8_OUTPUT = PolicyDecision(
    policy_bundle_sha256=sha("policybundle/v12"),
    allow=True, gate="HITL_REQUIRED",
    obligations=["co_sign:usr_cfinance", "sign:sha256-preimage", "audit_append:trace"],
    reason="spend_gate: 14200000 minor >= 5000000 minor -> human approval required; "
    "external_delivery requires co-signer per governance/hitl.yaml@v12",
    matched_rule="finance/spend_50k_co_sign",
)

S9_OUTPUT = SelfCorrectionEvent(
    run_id=RUN, step_id=6,
    trigger=ToolError(
        kind="SCHEMA_DRIFT",
        detail="Salesforce field shipment_event dropped from schema",
        observed_schema_version="sobject_shipment_event__r.v3",
        expected_schema_version="sobject_shipment_event__r.v2",
    ),
    strategy="FIELD_ALIAS_REMAP",
    mapping_detail={"from": "shipment_event.v3", "to": "v2 compat read", "confidence": 0.993},
    rows_revalidated_i64=142, outcome="RECOVERED", extra_latency_ms=870,
)


S10_INPUT = ApprovalRequest(
    approval_id="appr_01j8y71wze8q",
    run_id=RUN, step_id=4,
    title="PO-9407 renewal: Northwind $142,000/yr freight agreement",
    recommendation="approve",
    recommendation_basis="expiring clause §4 runs 8.2% over market; renewal caps surcharge — net -$21k/quarter",
    evidence={
        "pre_flight": ["duns+sanctions:ok", "budget_line:PDX-208 holds", "quotes:2 attached"],
        "cites": [SpanRef(doc_id="northwind_exp_2026", doc_version_sha256=sha("ver:northwind_exp/v1"),
                           locator="§4", quoted_text="base rate schedule Q4-2025").model_dump()],
    },
    blast_radius=[
        BlastRadiusItem(tool="docsign.envelope_send", effect="signs agreement — irreversible on counterparty signature", reversible="no"),
        BlastRadiusItem(tool="netsuite.po.create", effect="posts PO-9407 $142,000", reversible="with_counter_txn"),
        BlastRadiusItem(tool="slack.post.publish", effect="notifies #ops-apac", reversible="yes"),
    ],
    spend=Money(amount_minor=14_200_000, currency="USD"),
    policy_decision_sha256=sha("policydecision/S10"),
    required_signers=["usr_marcelino", "usr_cfinance"],
    min_signatures=2,
    created_at="2026-11-14T14:12:44Z",
    expires_at="2026-11-14T20:12:44Z",
)

S10_OUTPUT = ApprovalDecision(
    approval_id="appr_01j8y71wze8q",
    decision="APPROVE",
    signer=HUMAN,
    signature_sha256=sha("signature:marcelino-9407"),
    signed_at="2026-11-14T14:35:10Z",
    elapsed_since_request_ms=1_346_000,
)

S11_OUTPUT = MutationReceipt(
    receipt_id="rcpt_01j8y73xk2m0",
    approval_decision_sha256=sha("approvaldecision/S10"),
    tool="netsuite.po.create",
    external_id="PO-9407",
    state="posted",
    idempotency_key="idem_po9407_v1",
    observed_schema_version="netsuite_po.v9",
    mutated_at="2026-11-14T14:35:12Z",
)

S12_OUTPUT = TraceRow(
    seq=1_042,
    prev_hash_sha256=sha("genesis-trace-row"),
    ts="2026-11-14T14:05:40Z",
    run_id=RUN, step_id=2, actor=AGENT_AUDIT, op="ingest.parse",
    refs={"ingested_item": "sha256:a44d...e", "spans": "3", "acl": "4 scopes"},
    llm_ctx={"model_version": "qwen-doc-extract/v3.2", "prompt_sha256": sha("prompt/freight-plan-v6"), "temp": 0.0, "seed": 7},
    cost_usd_micro=21_400,
)

S13_OUTPUT = RoiAggRow(
    agent_id="agent_auditor_2",
    period="quarterly",
    started_at="2026-07-01T00:00:00Z",
    hours_offloaded_dec=5_056.0,
    runs_completed_i64=8_412,
    approval_latency_p50_s=1_320.0,
    hitl_acceptance_rate=0.92,
    selfcorrect_recovered_pct=88.4,
    gated_value_usd_minor_i64=184_000_00,
    run_cost_usd_micro_i64=96_000_000,
    policy_misses_i64=0,
)

_SAMPLES: dict[str, tuple[str, Any, str]] = {
    "S1_goal_intake": (S1_INPUT, S1_INPUT.model_dump_json, "input"),
    "S2_goal_spec": (S2_OUTPUT, S2_OUTPUT.model_dump_json, "output of parse LLM"),
    "S3_compiled_plan": (S3_OUTPUT, S3_OUTPUT.model_dump_json, "output of planner"),
    "S4_plan_validation": (S3_OUTPUT, None, "rules run over S3 — cycle/mutate/gate checks"),
    "S5_tool_request": (S5_INPUT, S5_INPUT.model_dump_json, "input to connector fabric"),
    "S5_tool_result": (S5_OUTPUT, S5_OUTPUT.model_dump_json, "output of connector fabric"),
    "S6_ingested_item": (S6_OUTPUT, S6_OUTPUT.model_dump_json, "output of doc-parse"),
    "S7_feature_vector": (S7_INPUT, S7_INPUT.model_dump_json, "input to anomaly models"),
    "S7_anomaly_score": (S7_OUTPUT, S7_OUTPUT.model_dump_json, "output of screen"),
    "S8_policy_input": (S8_INPUT, S8_INPUT.model_dump_json, "input to OPA"),
    "S8_policy_decision": (S8_OUTPUT, S8_OUTPUT.model_dump_json, "output of OPA"),
    "S9_selfcorrect_event": (S9_OUTPUT, S9_OUTPUT.model_dump_json, "emitted by executor"),
    "S10_approval_request": (S10_INPUT, S10_INPUT.model_dump_json, "emitted by gate"),
    "S10_approval_decision": (S10_OUTPUT, S10_OUTPUT.model_dump_json, "signed human output"),
    "S11_mutation_receipt": (S11_OUTPUT, S11_OUTPUT.model_dump_json, "connector after gated write"),
    "S12_trace_row": (S12_OUTPUT, S12_OUTPUT.model_dump_json, "append-only audit"),
    "S13_roi_row": (S13_OUTPUT, S13_OUTPUT.model_dump_json, "nightly aggregate"),
}


def main() -> int:
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name, (obj, dump, note) in _SAMPLES.items():
        try:
            payload = json.loads(dump()) if dump else {"validation": "passes on S3 payload"}
            # re-validate from serialized form — round-trip is part of the contract
            obj2 = type(obj).model_validate_json(obj.model_dump_json())
            assert obj2 == obj, "round-trip mismatch"
            out[name] = {"model": type(obj).__name__, "note": note, "sample": payload}
            print(f"ok   {name:28s} {type(obj).__name__}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            print(f"FAIL {name:28s} {exc}")
    if errors:
        return 1
    dest = HERE / "kurta_workflow_samples.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {dest} ({len(out)} validated step samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

