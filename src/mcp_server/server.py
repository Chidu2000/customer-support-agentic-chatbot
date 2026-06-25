"""FastMCP server exposing customer-support tools and chat."""

from mcp.server.fastmcp import FastMCP

from src.agents.graph import ask
from src.config import settings
from src.tools.support_tools import (
    get_customer_profile_and_tickets,
    get_support_ticket,
    ingest_policy_document,
    list_policy_sources,
    search_customer,
    search_policy_documents,
)
from src.vectorstore.chroma_store import add_documents

mcp = FastMCP(settings.mcp_server_name)


@mcp.tool()
def mcp_search_customer(name_or_email: str) -> str:
    """Search customers by name, email, or customer ID."""
    return search_customer.invoke({"name_or_email": name_or_email})


@mcp.tool()
def mcp_get_customer_profile_and_tickets(customer_id: int) -> str:
    """Get customer profile and support ticket history."""
    return get_customer_profile_and_tickets.invoke({"customer_id": customer_id})


@mcp.tool()
def mcp_get_support_ticket(ticket_id: int) -> str:
    """Get a support ticket by ID."""
    return get_support_ticket.invoke({"ticket_id": ticket_id})


@mcp.tool()
def mcp_search_policy_documents(query: str) -> str:
    """Search indexed company policy documents."""
    return search_policy_documents.invoke({"query": query})


@mcp.tool()
def mcp_ingest_policy_document(file_path: str) -> str:
    """Ingest a PDF or text policy file into the vector store."""
    return ingest_policy_document.invoke({"file_path": file_path})


@mcp.tool()
def mcp_list_policy_sources() -> str:
    """List indexed policy document sources."""
    return list_policy_sources.invoke({})


@mcp.tool()
def mcp_chat(question: str) -> str:
    """Ask the multi-agent support assistant a natural-language question."""
    result = ask(question)
    route = result.get("route", "unknown")
    answer = result.get("answer", "")
    return f"[route={route}]\n{answer}"


@mcp.resource("support://customers/sample")
def sample_customers_resource() -> str:
    """Read-only snapshot of a few customers for MCP clients."""
    from src.db import queries

    customers = queries.list_customers(limit=5)
    lines = [f"{c['id']}: {c['name']} <{c['email']}>" for c in customers[:5]]
    return "\n".join(lines) if lines else "No customers seeded."


if __name__ == "__main__":
    mcp.run(transport="stdio")
