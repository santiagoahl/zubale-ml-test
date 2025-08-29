# CLI: python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from src.agent_tools import json_reader, json_saver, action_plan_poster, yaml_saver
from src.io_schemas import ActionPlanModel

import argparse
import time
from dotenv import load_dotenv

load_dotenv()

llmops_callback_handler = CallbackHandler()

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
tools = [yaml_saver, json_reader, json_saver, action_plan_poster]
system_message = """
### Machine Learning Expert with focus on Data Drift

You are a helpul AI Assitant. Think step by step

Act as an expert in Machine Learning Ops with focus on data drift and model performance analysis.

Your goal is to emit an action plan with model quality status, findings and actions

### Action Plan Output

In order to deliver the action plan, you need to take in account that:

- Action plan is a JSON file (can be handled as dictionary for reasoning and acting)
- Action plan has the following schema

class ActionPlanModel(BaseModel):
    status: Literal["healthy", "warn", "critical"] = Field(..., description="Overall status of the monitored system")
    findings: List[str] = Field(default_factory=list, description="List of detected issues or observations")
    actions: List[Literal[
        "open_incident",
        "trigger_retraining",
        "roll_back_model",
        "raise_thresholds",
        "do_nothing"
    ]] = Field(default_factory=list, description="Actions to take based on findings")
    page_oncall: Optional[bool] = Field(default=False, description="Whether to page the on-call engineer")


### Tools

You have the following tools to accomplish your task

json_reader(path) -> Import a JSON file as dictionary from the given path. Use it to read the metrics and json
json_saver(path) -> Save a dictionary as JSON file in the given path. Use it to save the action plan
yaml_saver(path) -> Save the action plan as YML file in the given path (artifacts/). Use it to save the action plan
action_plan_poster(dict) -> Post the action plan (using the http method POST /monitor)

### Where is the data?

Both data/metrics_history.jsonl data/drift_latest.json


### Step-by-step suggested guide

1. Read the metrics data and classify status with the following euristics
 -  warn if ROC-AUC drops ≥ 3% vs 7-day median or p95 latency > 400ms for 2 consecutive
points.
 - critical if drop ≥ 6% or (overall_drift true and PR-AUC down ≥ 5%)
 - healthy otherwise

2. Analyze drift and metrics to report main findings (E.g. ROC-AUC or latency)

3. Identify actions to take (see the schema)

4. With the steps 1-3 build action plan (can be a dictionary), save it using the json_saver tool

5. Once you have the action plan as json, save the results
 - Save it as artifact in the artifacts/action_plan.yml dir using the yaml_saver tool
 - Post it using the action_plan_poster tool
""".strip()

react_agent = create_react_agent(llm, tools)


# Build ReAct Agent
def run_react_agent(metrics_path: str, drift_path: str, agent_plan_path: str) -> None:
    """
    Run Agentic AI ML Monitor

    Returns:
        None: Results are saved in the /monitor path of the API
    """
    user_query = f"""
    Analyse model quality

    Data
        - metrics history: {metrics_path}
        - drift report: {drift_path} 
        
    Save the agent plan in {agent_plan_path}"""

    time.sleep(60)  # Avoid TPM error
    react_agent.invoke(
        input={"messages": [SystemMessage(system_message), HumanMessage(user_query)]},
        config={"callbacks": [llmops_callback_handler]}
    )


def run_agent_monitor_cli():
    """
    Compute LLM Agent Analysis for action plan
    """
    # Read CLI params: path to input data and artifacts dir
    parser = argparse.ArgumentParser(description="Train Churn Model")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--drift", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()
    metrics_path, drift_path, agent_plan_path = (args.metrics, args.drift, args.out)

    # Run Agent
    run_react_agent(metrics_path, drift_path, agent_plan_path)


if __name__ == "__main__":
    run_agent_monitor_cli()
