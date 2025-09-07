import json
import pathlib

from urllib import request

# https://github.com/comfyanonymous/comfyui/blob/master/script_examples/basic_api_example.py


def load_workflow(filepath: pathlib.Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def comfyui_api_queue(workflow: dict):
    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


def test_workflow_runs():
    curdir = pathlib.Path(__file__).parent
    filepath = curdir / "../src/comfyui_olafrv/assets/" / "ORv_Google_Embedding_Gemma3_API.json"
    workflow = load_workflow(filepath)
    workflow["28"]["inputs"]["text"] = "woman in a lagoon"
    comfyui_api_queue(workflow)
