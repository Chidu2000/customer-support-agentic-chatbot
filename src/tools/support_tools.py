from pathlib import Path

from langchain_core.tools import tool

from src.db import queries
from src.ingestion.pdf_processor import ingest_file
from src.vectorstore.chroma_store import add_documents, list_indexed_sources, search_policies


@tool
def search_customer(name_or_email: str) -> str:
    """Search customers by name, email, or numeric customer ID."""
    results = queries.search_customers(name_or_email)
    if not results:
        return f"No customers found matching '{name_or_email}'."
    lines = []
    for customer in results:
        lines.append(
            f"ID {customer['id']}: {customer['name']} | {customer['email']} | "
            f"tier={customer.get('tier')} | status={customer.get('status')}"
        )
    return "\n".join(lines)


@tool
def get_customer_profile_and_tickets(customer_id: int) -> str:
    """Return a customer profile and their recent support tickets by customer ID."""
    customer = queries.get_customer_by_id(customer_id)
    if not customer:
        return f"No customer found with ID {customer_id}."
    tickets = queries.get_customer_tickets(customer_id)
    return queries.format_customer_overview(customer, tickets)


@tool
def get_support_ticket(ticket_id: int) -> str:
    """Fetch a single support ticket with customer context by ticket ID."""
    ticket = queries.get_ticket_by_id(ticket_id)
    if not ticket:
        return f"No ticket found with ID {ticket_id}."
    return (
        f"Ticket #{ticket['id']} for {ticket['customer_name']} (customer ID {ticket['customer_id']})\n"
        f"Subject: {ticket['subject']}\n"
        f"Status: {ticket['status']} | Priority: {ticket.get('priority')}\n"
        f"Created: {ticket['created_at']} | Resolved: {ticket.get('resolved_at') or 'N/A'}\n"
        f"Description: {ticket.get('description') or 'N/A'}"
    )


@tool
def search_policy_documents(query: str) -> str:
    """Search uploaded company policy documents for relevant passages."""
    docs = search_policies(query, k=4)
    if not docs:
        return "No policy documents indexed yet. Upload policy PDFs or run the ingest script first."
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_info = f", page {page + 1}" if page is not None else ""
        blocks.append(f"[{idx}] Source: {source}{page_info}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


@tool
def ingest_policy_document(file_path: str) -> str:
    """Ingest a policy PDF or text file into the vector knowledge base."""
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    chunks = ingest_file(path)
    count = add_documents(chunks)
    return f"Ingested {count} chunks from {path.name}."


@tool
def list_policy_sources() -> str:
    """List policy document sources currently indexed in the knowledge base."""
    sources = list_indexed_sources()
    if not sources:
        return "No policy documents indexed yet."
    return "Indexed policy sources:\n" + "\n".join(f"- {source}" for source in sources)


SQL_TOOLS = [search_customer, get_customer_profile_and_tickets, get_support_ticket]
RAG_TOOLS = [search_policy_documents]
ALL_TOOLS = SQL_TOOLS + RAG_TOOLS + [ingest_policy_document, list_policy_sources]
