# Transactional Outbox Pattern

The Transactional Outbox pattern resolves the dual-write problem in distributed event-driven systems. When a service must update a local database and publish an event to a message broker (e.g. Apache Kafka or RabbitMQ), performing these two operations independently introduces race conditions and potential data loss if the broker or database fails mid-operation.

## How the Outbox Pattern Works

1. **Atomic Local Transaction**: Instead of publishing directly to an external message broker, the service inserts the domain entity update and writes an event record into an `outbox` table within the same local database transaction.
2. **Transaction Commit**: The database transaction commits atomically. Either both the entity state and the outbox event are saved, or neither is.
3. **Asynchronous Publisher**: A separate background process (or Change Data Capture tool such as Debezium) periodically reads unpublished records from the outbox table and publishes them to the message broker.
4. **Mark as Processed / Delete**: Once the message broker acknowledges receipt, the publisher marks the outbox record as published or removes it from the outbox table.

## Architectural Benefits

- **Guaranteed At-Least-Once Delivery**: Ensures events are never lost due to network outages or message broker downtime.
- **Strong Local Consistency**: Prevents inconsistent states where a database record is saved but the event is lost, or vice versa.
- **Decoupled Business Logic**: Application services focus on core domain logic without managing complex retry loops for external brokers during request handling.
