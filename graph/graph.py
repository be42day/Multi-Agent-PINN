from agents import (
    manager_agent,
    designer_agent,
    trainer_agent,
    reporter_agent
)

from langgraph.graph import StateGraph, START, END
from graph.state_models import PINNDesignState


def build_pinn_design_graph(llm_general, llm_designer, max_error):
    # Agents
    manager = manager_agent.ManagerAgent(llm=llm_general, max_error=max_error)
    designer = designer_agent.DesignerAgent(llm=llm_designer)
    trainer = trainer_agent.TrainerAgent()
    reporter = reporter_agent.ReporterAgent(llm=llm_designer)

    # Initialize the graph
    builder = StateGraph(PINNDesignState)

    # Nodes
    builder.add_node("manager", manager.run_with_state)
    builder.add_node("designer", designer.run_with_state)
    builder.add_node("trainer", trainer.run_with_state)
    builder.add_node("reporter", reporter.run_with_state)

    # Edges
    def check_status(state: dict) -> str:
        return state.dict().get("project_status", "unknown")
    
    builder.add_edge(START, "manager")
    builder.add_conditional_edges(
        "manager",
        check_status,
        {
            "Send problem": "designer",
            "Send architecture": "trainer",
            "Send initial losses": "designer",
            "Send loss types": "trainer",
            "Send work history": "reporter",
            "Refine PINN": "designer"
        }
    )
    builder.add_edge("designer", "manager")
    builder.add_edge("trainer", "manager")
    builder.add_edge("reporter", END)

    # Compile the graph
    graph = builder.compile()
    return graph
