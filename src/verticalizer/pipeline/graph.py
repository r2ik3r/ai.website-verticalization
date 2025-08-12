from typing import TypedDict
from langgraph.graph import StateGraph, END
from ..pipeline.nodes import train_from_labeled, infer as infer_nodes
from ..models.persistence import save_model, load_model
from ..models.calibration import ProbCalibrator
from ..pipeline.io import read_table, write_jsonl
from ..utils.taxonomy import load_taxonomy
from ..scripts.excel_to_training_csv import excel_to_training_csv


class PipeState(TypedDict):
    action: str
    geo: str
    excel_path: str
    labeled_csv: str
    groundtruth_json: str
    model_path: str | None
    calib_path: str | None
    output_path: str | None
    report_path: str | None
    seed_path: str | None
    unlabeled_path: str | None
    iterations: int


def node_convert_excel(state: PipeState) -> PipeState:
    """Converts Excel to labeled CSV and groundtruth JSON before training."""
    excel_to_training_csv(state["excel_path"], state["geo"], state["labeled_csv"], state["groundtruth_json"])
    return state


def node_train(state: PipeState) -> PipeState:
    df = read_table(state["labeled_csv"])
    bundle = train_from_labeled(df)
    save_model(bundle["model"], state["model_path"])
    bundle["cal"].save(state["calib_path"])
    state["action"] = "done"
    return state


def node_infer(state: PipeState) -> PipeState:
    df = read_table(state["labeled_csv"])
    model = load_model(state["model_path"])
    cal = ProbCalibrator.load(state["calib_path"])
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    out = infer_nodes(model, cal, classes, df, topk=26)  # default all Tier-1
    write_jsonl(state["output_path"], out)
    state["action"] = "done"
    return state


def build_graph():
    workflow = StateGraph(PipeState)
    workflow.add_node("convert_excel", node_convert_excel)
    workflow.add_node("train", node_train)
    workflow.add_node("infer", node_infer)

    workflow.set_entry_point("convert_excel")
    workflow.add_edge("convert_excel", "train")
    workflow.add_edge("train", "infer")
    workflow.add_edge("infer", END)
    return workflow.compile()
