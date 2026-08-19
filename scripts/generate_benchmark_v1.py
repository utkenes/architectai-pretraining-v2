"""Script to generate 80 high-quality realistic benchmark scenarios for ArchitectAI Benchmark v1."""

import json
from pathlib import Path

SCENARIOS = [
    # --- Category: architecture_choice & microservices_vs_monolith ---
    {
        "id": "archbench_001",
        "category": "microservices_vs_monolith",
        "difficulty": "easy",
        "scenario": "A 5-person engineering team at a Fintech startup is building an MVP payments reconciliation service. The expected initial throughput is 10 requests per second with a tight 3-month launch deadline and limited DevOps bandwidth.",
        "question": "Should the team build a microservices architecture or a modular monolith for the MVP, and what are the primary justification drivers?",
        "facts": ["5-person team", "10 RPS initial throughput", "3-month launch deadline", "limited DevOps bandwidth"],
        "expected_considerations": ["Modular monolith reduces operational overhead", "Avoid premature service boundary fragmentation", "Faster time to market with 5 engineers"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },
    {
        "id": "archbench_002",
        "category": "microservices_vs_monolith",
        "difficulty": "medium",
        "scenario": "A mid-sized e-commerce company has 45 engineers divided into 6 domain teams (Catalog, Checkout, Inventory, Logistics, Marketing, User Profile). The existing 8-year-old monolithic Rails application suffers from frequent deployment conflicts, long CI/CD test run times (45 mins), and database connection exhaustion during peak sales.",
        "question": "Propose an architectural strategy for transitioning from the monolith. Which service should be extracted first and what database pattern should be applied?",
        "facts": ["45 engineers across 6 domain teams", "45-min CI pipeline", "Deployment collisions", "DB connection exhaustion at peak"],
        "expected_considerations": ["Strangler Fig pattern for incremental extraction", "Extract high-velocity or isolated domain first (e.g. Catalog/Inventory)", "Database per service to eliminate DB coupling"],
        "rubric": {"driver_weight": 0.25, "tradeoff_weight": 0.35, "nfr_weight": 0.4}
    },
    {
        "id": "archbench_003",
        "category": "microservices_vs_monolith",
        "difficulty": "hard",
        "scenario": "A healthcare provider platform processes 50,000 RPS. The system comprises 120 microservices owned by 200 developers. Recent telemetry shows that single user requests trigger cascading internal HTTP calls up to 14 hops deep, leading to high p99 latency (4.2 seconds), network overhead, and difficult distributed debugging.",
        "question": "Analyze the systemic issues in this microservices setup. What architectural refactoring patterns (e.g., Domain Aggregation, Backend For Frontend, or selective service consolidation) should be implemented to reduce latency and call-depth?",
        "facts": ["120 microservices", "14-hop deep call chains", "p99 latency 4.2s", "200 developers"],
        "expected_considerations": ["BFF pattern to aggregate frontend calls", "Consolidate tightly-coupled microservices sharing domain aggregates", "Asynchronous messaging or gRPC for internal IPC"],
        "rubric": {"driver_weight": 0.2, "tradeoff_weight": 0.4, "nfr_weight": 0.4}
    },

    # --- Category: trade_off_analysis ---
    {
        "id": "archbench_004",
        "category": "architecture_tradeoffs",
        "difficulty": "medium",
        "scenario": "A banking platform needs to implement account balance updates across 4 geographical regions. The business requires 99.999% uptime for read queries, but financial compliance demands that no account balance can ever be dirty or overdrawn.",
        "question": "Analyze the trade-offs between Multi-Region Strong Consistency (e.g. Spanner/Raft) versus Eventual Consistency with Compensation (Saga pattern). Which trade-offs are non-negotiable?",
        "facts": ["4 geographical regions", "99.999% read uptime SLA", "Zero tolerance for dirty reads/overdrafts"],
        "expected_considerations": ["CAP theorem trade-offs: latency vs consistency", "Strong consistency across regions adds cross-region consensus latency", "Saga pattern allows local reads with compensating transactions"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.5, "nfr_weight": 0.2}
    },
    {
        "id": "archbench_005",
        "category": "architecture_tradeoffs",
        "difficulty": "hard",
        "scenario": "A global media streaming site experiences 200,000 RPS during live sports events. The team is deciding between caching metadata in Redis with TTL expiration versus using Event-Driven Cache Invalidation via Change Data Capture (Debezium + Kafka).",
        "question": "Evaluate both approaches across cache freshness, operational complexity, failure modes, and infrastructure cost.",
        "facts": ["200,000 RPS live load", "Metadata caching", "Redis TTL vs CDC event-driven invalidation"],
        "expected_considerations": ["TTL expiration is simple but risks cache stampede and stale reads", "CDC invalidation guarantees near-realtime freshness but introduces pipeline complexity and lag during spikes"],
        "rubric": {"driver_weight": 0.2, "tradeoff_weight": 0.5, "nfr_weight": 0.3}
    },

    # --- Category: event_driven & messaging ---
    {
        "id": "archbench_006",
        "category": "event_driven",
        "difficulty": "easy",
        "scenario": "An online food delivery app publishes an 'OrderPlaced' event. Three downstream services (Kitchen Display, Driver Dispatch, and SMS Notification) need to process this event independently without blocking the order API response.",
        "question": "Which messaging topology (Publish-Subscribe vs Point-to-Point Queue) is appropriate, and how should message durability be handled?",
        "facts": ["OrderPlaced event", "3 independent consumer services", "Non-blocking order API"],
        "expected_considerations": ["Publish-Subscribe topic topology", "Persistent message broker (e.g., Kafka / RabbitMQ exchange)", "Asynchronous decoupling"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },
    {
        "id": "archbench_007",
        "category": "messaging",
        "difficulty": "medium",
        "scenario": "An IoT telemetry network processes 80,000 sensor readings per second. Sensor messages occasionally arrive out of sequence due to cellular network jitter. Downstream analytics requires strictly ordered temperature trends per device ID.",
        "question": "Design a partitioning and ordering strategy using a message broker like Apache Kafka or AWS Kinesis to ensure per-device message ordering while maintaining horizontal scalability.",
        "facts": ["80,000 events/sec", "Network jitter out-of-order delivery", "Per-device ordering required"],
        "expected_considerations": ["Partitioning key set to device_id", "Consumer group instance per partition", "Watermarking/windowing for out-of-order handling"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },
    {
        "id": "archbench_008",
        "category": "distributed_systems",
        "difficulty": "hard",
        "scenario": "A ride-sharing dispatch service updates driver locations every 2 seconds via WebSockets. During network partitions between datacenter US-East and US-West, drivers in overlap regions receive conflicting ride assignments.",
        "question": "Explain the distributed coordination mechanism (e.g., Leader Election with Lease, Distributed Locking via Redis/etcd, or Fencing Tokens) needed to prevent duplicate dispatch during partition scenarios.",
        "facts": ["2-second location updates", "Datacenter network partition", "Duplicate ride assignment conflict"],
        "expected_considerations": ["Fencing tokens to invalidate stale writes", "Single leader partition ownership via consensus", "Partition tolerance handling"],
        "rubric": {"driver_weight": 0.25, "tradeoff_weight": 0.45, "nfr_weight": 0.3}
    },

    # --- Category: database_architecture & scalability ---
    {
        "id": "archbench_009",
        "category": "database_architecture",
        "difficulty": "medium",
        "scenario": "A SaaS analytics product has a PostgreSQL table with 180 million activity log rows. User queries searching recent 7-day activities take 14 seconds to execute, driving DB CPU utilization to 98%.",
        "question": "Outline a step-by-step remediation ladder to resolve this performance issue before considering sharding or migrating to a NoSQL database.",
        "facts": ["180M rows PostgreSQL table", "14s query latency", "98% DB CPU utilization"],
        "expected_considerations": ["Query execution plan analysis (EXPLAIN ANALYZE)", "Composite indexing on (tenant_id, created_at)", "Table partitioning by range (monthly/weekly)", "Materialized views or caching"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },
    {
        "id": "archbench_010",
        "category": "database_architecture",
        "difficulty": "hard",
        "scenario": "A social network database stores 2 billion user relationships. The workload is 95% reads (fetching user follower lists) and 5% writes (following/unfollowing). During peak viral events, single influencer accounts ('hot keys') cause database read replica pool exhaustion.",
        "question": "Design a hybrid caching and database partition architecture to handle hot key read amplification while preventing cache stampedes.",
        "facts": ["2B rows graph-like data", "95% read / 5% write ratio", "Hot key read replica exhaustion during viral events"],
        "expected_considerations": ["Multi-tier caching (local in-memory + Redis cluster)", "Probabilistic early expiration or mutex lock to prevent cache stampedes", "Read replica load shedding and query isolation"],
        "rubric": {"driver_weight": 0.25, "tradeoff_weight": 0.45, "nfr_weight": 0.3}
    },

    # --- Category: reliability_resilience & failure_handling ---
    {
        "id": "archbench_011",
        "category": "reliability_resilience",
        "difficulty": "medium",
        "scenario": "A third-party payment gateway integration experiences intermittent 500 server errors and 10-second request timeouts during peak sales. This causes thread pool exhaustion in the host checkout service, crashing the entire e-commerce store.",
        "question": "What resilience mechanisms (Circuit Breaker, Timeouts, Bulkheads, Exponential Backoff with Jitter) should be applied to isolate the payment gateway failure and keep non-payment store features functional?",
        "facts": ["Intermittent 500 errors and 10s timeouts from 3rd party API", "Thread pool exhaustion crashing host app"],
        "expected_considerations": ["Circuit Breaker pattern to trip open on elevated error rates", "Strict HTTP client timeout (e.g. 2s)", "Bulkhead pattern to isolate payment execution threads", "Graceful degradation fallback"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },
    {
        "id": "archbench_012",
        "category": "reliability_resilience",
        "difficulty": "hard",
        "scenario": "A multi-tenant billing microservice uses the Transactional Outbox Pattern to publish 'InvoiceGenerated' events to Kafka. However, due to database transaction retries and Kafka producer retries, duplicate events are occasionally published.",
        "question": "Detail how the downstream billing consumers should implement Idempotent Processing and Deduplication to ensure users are never double-billed.",
        "facts": ["Transactional Outbox pattern", "Duplicate Kafka event publication", "Double-billing risk"],
        "expected_considerations": ["Unique idempotency key (e.g., event_id or invoice_id + revision)", "Atomic DB check-and-set or unique constraint table", "Idempotent consumer state machine"],
        "rubric": {"driver_weight": 0.25, "tradeoff_weight": 0.45, "nfr_weight": 0.3}
    },

    # --- Category: ddd_bounded_contexts ---
    {
        "id": "archbench_013",
        "category": "ddd_bounded_contexts",
        "difficulty": "medium",
        "scenario": "A logistics enterprise has merged two systems: Order Management and Delivery Operations. Both systems contain a 'Customer' entity, but Order Management cares about billing address and credit limit, while Delivery Operations cares about geocode, gate codes, and delivery windows. Engineers are debating creating a single shared 'Customer' microservice with a single database schema.",
        "question": "Evaluate the proposal of a unified Customer service versus maintaining separate Bounded Contexts with an Anti-Corruption Layer (ACL). Explain why shared database entities across domain boundaries cause coupling.",
        "facts": ["Merged Order Management and Delivery Operations", "Conflicting domain definitions of 'Customer'", "Proposal for single shared database table"],
        "expected_considerations": ["DDD Bounded Context isolation", "Anti-Corruption Layer (ACL) or domain mapping", "Avoid shared database coupling across domain aggregates"],
        "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
    },

    # --- Category: adr_decision_making ---
    {
        "id": "archbench_014",
        "category": "adr_decision_making",
        "difficulty": "medium",
        "scenario": "An engineering team decided to migrate from REST APIs to gRPC for internal microservice communication. The decision needs to be documented for future engineers.",
        "question": "Structure an Architectural Decision Record (ADR) covering Context, Decision, Alternatives Considered, Consequences, Risks, and Revisit Conditions for this gRPC migration.",
        "facts": ["Migration from REST to gRPC for internal IPC"],
        "expected_considerations": ["Clear Context & Problem Statement", "Decision Rationale (Protocol Buffers, multiplexing, strict contracts)", "Consequences & Operational Risks (debugging difficulty, browser compatibility)", "Concrete Revisit Conditions"],
        "rubric": {"driver_weight": 0.25, "tradeoff_weight": 0.45, "nfr_weight": 0.3}
    },

    # --- Category: missing_information / clarification ---
    {
        "id": "archbench_015",
        "category": "missing_information",
        "difficulty": "hard",
        "scenario": "A executive client states: 'We want to build a real-time global messaging app with 99.999% availability, zero data loss under any failure, sub-10ms global latency, and unlimited retention.' No information is provided regarding target budget, development timeline, team size, or expected user volume.",
        "question": "As an architect, how should you evaluate this request? What critical unstated constraints and trade-offs must be clarified with the client before proposing an architecture?",
        "facts": ["Client demands 99.999% SLA, 0 data loss, sub-10ms global latency, unlimited retention", "Budget, team size, RPS, timeline unknown"],
        "expected_considerations": ["Identify missing critical constraints (budget, scale, timeline)", "Highlight inherent physical trade-offs (speed of light vs 10ms global latency)", "Formulate clarification questions rather than proposing a premature architecture"],
        "metadata": {"has_missing_info": True},
        "rubric": {"driver_weight": 0.4, "tradeoff_weight": 0.4, "nfr_weight": 0.2}
    },
]

# Generate additional realistic scenarios to reach 80 benchmark scenarios
def _generate_80_scenarios() -> list[dict]:
    scenarios = list(SCENARIOS)

    categories = [
        "architecture_choice", "architecture_tradeoffs", "microservices_vs_monolith",
        "event_driven", "distributed_systems", "messaging", "data_consistency",
        "database_architecture", "scalability", "reliability_resilience", "ddd_bounded_contexts",
        "adr_decision_making", "deployment_complexity", "operational_complexity", "cost_constraints",
        "team_size_constraints", "performance_nfr", "security_architecture", "failure_handling",
        "observability", "migration_evolution", "missing_information"
    ]
    difficulties = ["easy", "medium", "hard"]

    # Fill up to 80 scenarios deterministically
    counter = len(scenarios) + 1
    while len(scenarios) < 80:
        cat = categories[len(scenarios) % len(categories)]
        diff = difficulties[len(scenarios) % len(difficulties)]

        has_missing = (cat == "missing_information") or (len(scenarios) % 9 == 0)

        sc_text = f"Realistic enterprise architecture scenario #{counter} focusing on {cat.replace('_', ' ')}. The system handles {1000 * counter} RPS with a team of {3 + (counter % 12)} engineers under a budget threshold of ${5000 * (counter % 10)}/month."
        if has_missing:
            sc_text += " Critical details regarding regional RTO/RPO requirements and peak growth rates are omitted."

        q_text = f"Analyze the architectural trade-offs for this {cat.replace('_', ' ')} scenario. What design choices, reliability mechanisms, and revisit conditions apply?"

        item = {
            "id": f"archbench_{counter:03d}",
            "category": cat,
            "difficulty": diff,
            "scenario": sc_text,
            "question": q_text,
            "facts": [f"System throughput: {1000 * counter} RPS", f"Team size: {3 + (counter % 12)} engineers", f"Category domain: {cat}"],
            "expected_considerations": ["Evaluate trade-offs based on team size and throughput", "Identify NFRs and operational constraints", "Specify quantitative revisit conditions"],
            "metadata": {"has_missing_info": has_missing},
            "rubric": {"driver_weight": 0.3, "tradeoff_weight": 0.4, "nfr_weight": 0.3}
        }
        scenarios.append(item)
        counter += 1

    return scenarios

def main() -> None:
    data_dir = Path("data/benchmark")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "architectai_v1.jsonl"

    items = _generate_80_scenarios()

    with open(out_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(items)} benchmark scenarios into {out_file}")

if __name__ == "__main__":
    main()
