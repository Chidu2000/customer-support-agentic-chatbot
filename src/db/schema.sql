CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    tier TEXT DEFAULT 'standard',
    status TEXT DEFAULT 'active',
    joined_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS support_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
