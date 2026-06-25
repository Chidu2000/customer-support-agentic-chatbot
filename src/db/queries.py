from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_engine


def list_customers(limit: int = 10, engine: Engine | None = None) -> list[dict]:
    engine = engine or get_engine()
    sql = text(
        """
        SELECT id, name, email, phone, tier, status, joined_at, notes
        FROM customers
        ORDER BY name
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
    return [dict(row) for row in rows]


def search_customers(query: str, engine: Engine | None = None) -> list[dict]:
    engine = engine or get_engine()
    pattern = f"%{query.strip()}%"
    sql = text(
        """
        SELECT id, name, email, phone, tier, status, joined_at, notes
        FROM customers
        WHERE name LIKE :pattern
           OR email LIKE :pattern
           OR CAST(id AS TEXT) = :exact
        ORDER BY name
        LIMIT 10
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"pattern": pattern, "exact": query.strip()}).mappings().all()
    return [dict(row) for row in rows]


def get_customer_by_id(customer_id: int, engine: Engine | None = None) -> dict | None:
    engine = engine or get_engine()
    sql = text(
        """
        SELECT id, name, email, phone, tier, status, joined_at, notes
        FROM customers
        WHERE id = :customer_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"customer_id": customer_id}).mappings().first()
    return dict(row) if row else None


def get_customer_tickets(customer_id: int, engine: Engine | None = None) -> list[dict]:
    engine = engine or get_engine()
    sql = text(
        """
        SELECT id, customer_id, subject, description, status, priority, created_at, resolved_at
        FROM support_tickets
        WHERE customer_id = :customer_id
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"customer_id": customer_id}).mappings().all()
    return [dict(row) for row in rows]


def get_ticket_by_id(ticket_id: int, engine: Engine | None = None) -> dict | None:
    engine = engine or get_engine()
    sql = text(
        """
        SELECT t.id, t.customer_id, c.name AS customer_name, t.subject, t.description,
               t.status, t.priority, t.created_at, t.resolved_at
        FROM support_tickets t
        JOIN customers c ON c.id = t.customer_id
        WHERE t.id = :ticket_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"ticket_id": ticket_id}).mappings().first()
    return dict(row) if row else None


def format_customer_overview(customer: dict, tickets: list[dict]) -> str:
    lines = [
        f"Customer: {customer['name']} (ID {customer['id']})",
        f"Email: {customer['email']}",
        f"Phone: {customer.get('phone') or 'N/A'}",
        f"Tier: {customer.get('tier', 'standard')} | Status: {customer.get('status', 'active')}",
        f"Joined: {customer.get('joined_at', 'N/A')}",
    ]
    if customer.get("notes"):
        lines.append(f"Notes: {customer['notes']}")

    lines.append("")
    if not tickets:
        lines.append("Support tickets: none on record.")
    else:
        lines.append(f"Support tickets ({len(tickets)}):")
        for ticket in tickets:
            lines.append(
                f"- #{ticket['id']} [{ticket['status']}/{ticket.get('priority', 'medium')}] "
                f"{ticket['subject']} (created {ticket['created_at']})"
            )
            if ticket.get("description"):
                lines.append(f"  {ticket['description']}")
    return "\n".join(lines)
