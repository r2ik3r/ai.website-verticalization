from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from ..pipeline.nodes import train_from_labeled, infer as infer_nodes, evaluate as eval_nodes
from ..pipeline.self_train import self_training_loop
from ..models.persistence import save_model, load_model
from ..models.calibration import ProbCalibrator
from ..pipeline.io import read_table

class PipeState(TypedDict):
    action: str
    geo: str
    input_path: str
    model_path: str | None
    calib_path: str | None
    output_path: str | None
    report_path: str | None
    seed_path: str | None
    unlabeled_path: str | None
    iterations: int

def node_train(state: PipeState) -> PipeState:
    df = read_table(state["input_path"])
    bundle = train_from_labeled(df)
    save_model(bundle["model"], state["model_path"])
    bundle["cal"].save(state["calib_path"])
    state["action"] = "done"
    return state

def node_self_train(state: PipeState) -> PipeState:
    import pandas as pd
    seed_df = read_table(state["seed_path"])
    unlabeled_df = read_table(state["unlabeled_path"])
    model, cal, classes = self_training_loop(seed_df, unlabeled_df, iterations=state["iterations"])
    save_model(model, state["model_path"])
    cal.save(state["calib_path"])
    state["action"] = "done"
    return state

def node_infer(state: PipeState) -> PipeState:
    import json, orjson
    from ..pipeline.io import write_jsonl
    df = read_table(state["input_path"])
    model = load_model(state["model_path"])
    cal = ProbCalibrator.load(state["calib_path"]) if state["calib_path"] else ProbCalibrator()
    # classes are Tier-1 IAB in order
    from ..utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    out = infer_nodes(model, cal, classes, df)
    write_jsonl(state["output_path"], out)
    state["action"] = "done"
    return state

def build_graph():
    g = StateGraph(PipeState)
    g.add_node("train", node_train)
    g.add_node("self_train", node_self_train)
    g.add_node("infer", node_infer)
    def route(state: PipeState):
        if state["action"] in {"train"}:
            return "train"
        if state["action"] in {"self_train"}:
            return "self_train"
        if state["action"] in {"infer"}:
            return "infer"
        return END
    g.set_entry_point("train")
    g.add_conditional_edges("train", lambda s: "infer" if s["action"] == "infer" else END, {"infer": "infer", "end": END})
    # compile with flexible entry using small wrapper
    graph = g.compile()
    return graph
