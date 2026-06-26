import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from src.llm.factory import get_chat_model
from src.tools.support_tools import RAG_TOOLS, SQL_TOOLS
from src.vectorstore.chroma_store import search_policies

Route = Literal["sql", "rag", "hybrid"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    route: Route
    sql_context: str
    rag_context: str
    final_answer: str
    rag_sources: list[dict]  # [{source_file, page}] for citation display


ROUTER_PROMPT = """You route customer-support questions for a multi-agent assistant.

Routes:
- sql   → question is ONLY about a specific customer, their profile, account, plan, or support tickets.
          Examples: "Show me Ema's profile", "What tickets does Marcus have?"

- rag   → question is ONLY about a company policy, rule, or document (no specific customer mentioned).
          Examples: "What is the refund policy?", "How does shipping work?"

- hybrid → question mentions a specific customer/person AND also asks about a policy, plan, or eligibility.
           ALWAYS choose hybrid when a customer name appears alongside a policy topic.
           Examples:
             "Priya wants to know about the warranty policy for her plan"
             "Can Ema get a refund for her open ticket?"
             "Does Marcus qualify for express shipping?"
             "What plan is David on and what are the cancellation terms?"

When in doubt between rag and hybrid, choose hybrid if any customer name or account is mentioned.

Reply with only one word: sql, rag, or hybrid."""


SQL_AGENT_PROMPT = """You are a customer data specialist for a support team.
Use the available tools to retrieve accurate customer and ticket information.
Summarize findings clearly for a support executive. Do not invent data."""


RAG_AGENT_PROMPT = """You are a policy knowledge specialist.
Use search_policy_documents to retrieve relevant policy text before answering.
Answer only from retrieved policy content. If policies are missing, say so clearly."""


SYNTHESIS_PROMPT = """You are a senior customer support assistant.
Combine structured customer data and policy excerpts into one concise, helpful answer.
Be accurate, friendly, and actionable. Do not invent facts beyond the provided context."""


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


_POLICY_KEYWORDS = {
    "policy", "refund", "return", "shipping", "warranty", "privacy",
    "terms", "cancellation", "guarantee", "coverage", "plan", "eligible",
    "eligib", "qualify", "entitle",
}

_CUSTOMER_SIGNALS = {
    "customer", "client", "user", "account", "she", "he", "they",
    "her", "his", "their", "ticket", "profile",
}


def _has_named_entity(text: str) -> bool:
    """Heuristic: any capitalised word that isn't sentence-start is likely a name."""
    words = text.split()
    # A word is a candidate name if it starts with a capital and is not the first word
    return any(w[0].isupper() and i > 0 for i, w in enumerate(words) if w.isalpha())


def _rule_based_route(question: str) -> Route | None:
    """Safety-net: override LLM if the pattern is unambiguous."""
    q = question.lower()
    has_policy = any(kw in q for kw in _POLICY_KEYWORDS)
    has_customer = any(kw in q for kw in _CUSTOMER_SIGNALS) or _has_named_entity(question)

    if has_policy and has_customer:
        return "hybrid"
    return None  # let LLM decision stand


def route_query(state: AgentState) -> AgentState:
    llm = get_chat_model(temperature=0)
    question = _latest_user_text(state["messages"])

    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=question),
        ]
    )
    route_text = str(response.content).strip().lower()

    if "hybrid" in route_text:
        llm_route: Route = "hybrid"
    elif "rag" in route_text:
        llm_route = "rag"
    elif "sql" in route_text:
        llm_route = "sql"
    else:
        llm_route = "rag" if any(kw in question.lower() for kw in _POLICY_KEYWORDS) else "sql"

    # Rule-based override catches cases where the LLM under-routes to rag/sql
    route = _rule_based_route(question) or llm_route
    return {"route": route}


def run_sql_agent(state: AgentState) -> AgentState:
    llm = get_chat_model()
    agent = create_react_agent(llm, SQL_TOOLS, prompt=SystemMessage(content=SQL_AGENT_PROMPT))
    result = agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    return {
        "sql_context": str(answer),
        "messages": [AIMessage(content=str(answer))],
    }


def _collect_rag_sources(question: str) -> list[dict]:
    """Return deduplicated source citations for the retrieved chunks."""
    docs = search_policies(question, k=4)
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in docs:
        source_file = doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source_file": source_file, "page": page})
    return sources


def run_rag_agent(state: AgentState) -> AgentState:
    llm = get_chat_model()
    agent = create_react_agent(llm, RAG_TOOLS, prompt=SystemMessage(content=RAG_AGENT_PROMPT))
    result = agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    question = _latest_user_text(state["messages"])
    return {
        "rag_context": str(answer),
        "rag_sources": _collect_rag_sources(question),
        "messages": [AIMessage(content=str(answer))],
    }


def run_hybrid_agents(state: AgentState) -> AgentState:
    llm = get_chat_model()
    sql_agent = create_react_agent(llm, SQL_TOOLS, prompt=SystemMessage(content=SQL_AGENT_PROMPT))
    rag_agent = create_react_agent(llm, RAG_TOOLS, prompt=SystemMessage(content=RAG_AGENT_PROMPT))

    sql_result = sql_agent.invoke({"messages": state["messages"]})
    rag_result = rag_agent.invoke({"messages": state["messages"]})

    sql_context = str(sql_result["messages"][-1].content)
    rag_context = str(rag_result["messages"][-1].content)
    rag_sources = _collect_rag_sources(_latest_user_text(state["messages"]))

    synthesis = llm.invoke(
        [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{_latest_user_text(state['messages'])}\n\n"
                    f"Customer/ticket findings:\n{sql_context}\n\n"
                    f"Policy findings:\n{rag_context}"
                )
            ),
        ]
    )
    final_answer = str(synthesis.content)
    return {
        "sql_context": sql_context,
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }


def finalize_single_agent(state: AgentState) -> AgentState:
    if state.get("final_answer"):
        return {}
    latest = state["messages"][-1].content if state.get("messages") else ""
    return {"final_answer": str(latest)}


def pick_route(state: AgentState) -> str:
    route = state.get("route", "sql")
    if route == "hybrid":
        return "hybrid"
    if route == "rag":
        return "rag"
    return "sql"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", route_query)
    graph.add_node("sql_agent", run_sql_agent)
    graph.add_node("rag_agent", run_rag_agent)
    graph.add_node("hybrid_agent", run_hybrid_agents)
    graph.add_node("finalize", finalize_single_agent)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        pick_route,
        {
            "sql": "sql_agent",
            "rag": "rag_agent",
            "hybrid": "hybrid_agent",
        },
    )
    graph.add_edge("sql_agent", "finalize")
    graph.add_edge("rag_agent", "finalize")
    graph.add_edge("hybrid_agent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def ask(question: str, history: list[BaseMessage] | None = None) -> dict:
    graph = build_graph()
    messages = list(history or []) + [HumanMessage(content=question)]
    result = graph.invoke(
        {
            "messages": messages,
            "route": "sql",
            "sql_context": "",
            "rag_context": "",
            "rag_sources": [],
            "final_answer": "",
        }
    )
    return {
        "answer": result.get("final_answer") or str(result["messages"][-1].content),
        "route": result.get("route", "sql"),
        "sql_context": result.get("sql_context", ""),
        "rag_context": result.get("rag_context", ""),
        "rag_sources": result.get("rag_sources", []),
    }
