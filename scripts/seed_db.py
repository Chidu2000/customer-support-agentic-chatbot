"""Seed synthetic customer and support ticket data."""

from sqlalchemy import text

from src.db.connection import get_engine, init_schema


CUSTOMERS = [
    {
        "name": "Ema Johnson",
        "email": "ema.johnson@example.com",
        "phone": "+1-555-0101",
        "tier": "gold",
        "status": "active",
        "joined_at": "2022-03-14",
        "notes": "Prefers email contact. Loyal customer since 2022.",
    },
    {
        "name": "Marcus Lee",
        "email": "marcus.lee@example.com",
        "phone": "+1-555-0102",
        "tier": "standard",
        "status": "active",
        "joined_at": "2023-07-02",
        "notes": "Requested faster shipping on last order.",
    },
    {
        "name": "Priya Sharma",
        "email": "priya.sharma@example.com",
        "phone": "+1-555-0103",
        "tier": "platinum",
        "status": "active",
        "joined_at": "2021-11-20",
        "notes": "Enterprise account contact.",
    },
    {
        "name": "David Chen",
        "email": "david.chen@example.com",
        "phone": "+1-555-0104",
        "tier": "standard",
        "status": "inactive",
        "joined_at": "2020-05-09",
        "notes": "Account paused due to payment issue.",
    },
]

TICKETS = [
    (1, "Refund request for order #8821", "Customer reports duplicate charge and wants refund.", "open", "high", "2025-06-01", None),
    (1, "Shipping delay inquiry", "Package stuck in transit for 5 days.", "resolved", "medium", "2025-05-12", "2025-05-15"),
    (1, "Product warranty question", "Asked about extended warranty for laptop purchase.", "resolved", "low", "2025-04-03", "2025-04-04"),
    (2, "Login issues", "Cannot reset password via mobile app.", "open", "medium", "2025-06-10", None),
    (2, "Billing address update", "Needs billing address changed before next invoice.", "resolved", "low", "2025-05-20", "2025-05-21"),
    (3, "Bulk order discount", "Negotiating pricing for 500-unit order.", "in_progress", "high", "2025-06-08", None),
    (3, "API integration support", "Webhook failures on order status updates.", "open", "high", "2025-06-15", None),
    (4, "Account reactivation", "Wants to reactivate after resolving payment.", "closed", "medium", "2025-03-01", "2025-03-05"),
]


def seed() -> None:
    engine = get_engine()
    init_schema(engine)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM support_tickets"))
        conn.execute(text("DELETE FROM customers"))

        for customer in CUSTOMERS:
            conn.execute(
                text(
                    """
                    INSERT INTO customers (name, email, phone, tier, status, joined_at, notes)
                    VALUES (:name, :email, :phone, :tier, :status, :joined_at, :notes)
                    """
                ),
                customer,
            )

        for ticket in TICKETS:
            conn.execute(
                text(
                    """
                    INSERT INTO support_tickets
                    (customer_id, subject, description, status, priority, created_at, resolved_at)
                    VALUES (:customer_id, :subject, :description, :status, :priority, :created_at, :resolved_at)
                    """
                ),
                {
                    "customer_id": ticket[0],
                    "subject": ticket[1],
                    "description": ticket[2],
                    "status": ticket[3],
                    "priority": ticket[4],
                    "created_at": ticket[5],
                    "resolved_at": ticket[6],
                },
            )

    print("Database seeded with synthetic customer and ticket data.")


if __name__ == "__main__":
    seed()
