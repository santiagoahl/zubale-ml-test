import yaml
import json
from pathlib import Path
from src.io_schemas import ActionPlanModel
from src.app import post_action_plan



import json
from pathlib import Path


def json_reader(file_path: str) -> dict | list:
    """
    Read a JSON file and return its content as a dictionary or a list of dictionaries.

    Parameters
    ----------
    file_path : str
        Path to the JSON file to read.

    Returns
    -------
    dict | list
        Dictionary with the contents of the JSON file or a list of dictionaries if it's a JSONL file.

    Example
    -------
    >>> json_reader("artifacts/drift_report.json")
    {"status": "healthy", "findings": [...], "actions": ["do_nothing"]}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # Attempt to load as a standard JSON object.
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # If it fails, check for the specific 'Extra data' error which indicates JSONL.
            if "Extra data" in str(e):
                # Process as JSONL: parse each line as a separate JSON object.
                return [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                # Re-raise the error if it's a different JSON decoding issue.
                raise e 


def json_saver(data: dict, file_path: str, indent: int = 4) -> None:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    file_path : str
        Path to the output JSON file.
    indent : int, optional
        Number of spaces to use as indentation (default is 4).

    Returns
    -------
    None

    Example
    -------
    >>> json_saver({"status": "warn"}, "artifacts/action_plan.json")
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def yaml_saver(data: dict, file_path: str, default_flow_style: bool = False) -> None:
    """
    Save a dictionary as a YAML file.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    file_path : str
        Path to the output YAML file.
    default_flow_style : bool, optional
        Whether to use block style (False) or flow style (True) in YAML (default is False).

    Returns
    -------
    None

    Example
    -------
    >>> yaml_saver({"status": "warn"}, "artifacts/action_plan.yaml")
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=False)


def action_plan_poster(action_plan: ActionPlanModel) -> dict:
    """
    Post to the /monitor endpoint the action plan resource.

    Parameters
    ----------
    action_plan : ActionPlanModel
        Dictionary with keys: status, findings and actions.

    Returns
    -------
    dict
        Response from the endpoint (the posted action plan).

    Example
    -------
    >>> action_plan_poster({
            "status": "healthy",
            "findings": ["ROC-AUC = 0.9", "p95 latency > 400ms for 2 consecutive points"],
            "actions": ["do_nothing"],
        })
    """
    return post_action_plan(action_plan)
