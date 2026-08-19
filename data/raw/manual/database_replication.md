# Database Replication Strategies

Database replication is the technique of sharing information across multiple database servers to ensure fault tolerance, high availability, and improved read performance.

## Primary Replication Topologies

### 1. Single-Leader (Primary-Replica) Replication
- All writes are directed to a single primary leader node.
- The leader writes data locally and streams changes via a write-ahead log (WAL) or replication stream to one or more replica nodes.
- Replicas serve read-only queries, offloading read volume from the primary leader.

### 2. Multi-Leader Replication
- Multiple database nodes accept write operations simultaneously.
- Each leader acts as a replica for other leaders.
- Commonly used across multi-region datacenters to provide low-write latency, requiring conflict resolution strategies (e.g. Last-Write-Wins or Operational Transformation).

### 3. Leaderless (Dynamo-Style) Replication
- Any node can accept read and write requests directly from clients.
- Uses quorum consensus rules ($R + W > N$) to guarantee that read sets and write sets overlap, ensuring clients observe up-to-date data.

## Synchronous vs. Asynchronous Replication

- **Synchronous Replication**: The primary waits for replicas to confirm disk write before returning success to the client. Ensures zero data loss (RPO = 0) but increases write latency.
- **Asynchronous Replication**: The primary returns success immediately after writing locally. Highly performant, but carries a risk of data loss if the leader crashes before changes propagate.
