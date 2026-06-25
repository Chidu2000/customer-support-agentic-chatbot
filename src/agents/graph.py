import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from src.llm.factory import get_chat_model
from src.tools.support_tools import RAG_TOOLS, SQL_TOOLS

Route = Literal["sql", "rag", "hybrid"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    route: Route
    sql_context: str
    rag_context: str
    final_answer: str


ROUTER_PROMPT = """You route customer-support questions for a multi-agent assistant.

Choose exactly one route:
- sql: questions about customer profiles, support tickets, account details, ticket status
- rag: questions about company policies, refunds, shipping, privacy, terms, uploaded documents
- hybrid: questions needing BOTH customer/ticket data AND policy/document context

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
        route: Route = "hybrid"
    elif "rag" in route_text or "policy" in route_text:
        route = "rag"
    elif "sql" in route_text or "customer" in route_text or "ticket" in route_text:
        route = "sql"
    else:
        route = "rag" if any(word in question.lower() for word in ("policy", "refund", "shipping", "privacy")) else "sql"
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


def run_rag_agent(state: AgentState) -> AgentState:
    llm = get_chat_model()
    agent = create_react_agent(llm, RAG_TOOLS, prompt=SystemMessage(content=RAG_AGENT_PROMPT))
    result = agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    return {
        "rag_context": str(answer),
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
            "final_answer": "",
        }
    )
    return {
        "answer": result.get("final_answer") or str(result["messages"][-1].content),
        "route": result.get("route", "sql"),
        "sql_context": result.get("sql_context", ""),
        "rag_context": result.get("rag_context", ""),
    }
