"""Use the pinned, separately obtained upstream DCGen segmentation classes.

Upstream source is not redistributed by this package. Only the two required
class definitions are loaded; unrelated browser and provider modules are not.
"""
from __future__ import annotations
import ast
import hashlib
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import release_paths

@lru_cache(maxsize=1)
def _upstream_class():
    path = release_paths.DCGEN / "utils.py"
    if not path.is_file():
        raise FileNotFoundError("Set DCGEN_HOME to the pinned DCGen checkout; see docs/RUNTIME.md")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name in {"ImgNode", "ImgSegmentation"}]
    if [node.name for node in selected] != ["ImgNode", "ImgSegmentation"]:
        raise ValueError("Upstream DCGen segmentation classes are missing")
    expected = ["18d940a3b01562c7ebb97c8209397e1e8e59c87827d3be3468642a505e6a13a0",
                "ba1ba5ba95a2cda5118d0ef25f2e569444d907fb5ac9a5c81b56b0bcf2a323f7"]
    actual = [hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
              for node in selected]
    if actual != expected:
        raise ValueError("Unexpected DCGen source; use the version documented in docs/RUNTIME.md")
    namespace = dict(ABC=ABC, abstractmethod=abstractmethod, Union=Union,
                     Image=Image, ImageDraw=ImageDraw, ImageEnhance=ImageEnhance,
                     np=np, json=json)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ImgSegmentation"]

class UpstreamImgSegmentation:
    """Construct the original upstream class without copying its source."""
    def __new__(cls, image, *args, **kwargs):
        if isinstance(image, Path):
            image = str(image)
        return _upstream_class()(image, *args, **kwargs)

def segment_screenshot(image_path, max_depth=2, var_thresh=50, diff_thresh=45,
                       diff_portion=0.9, window_size=50):
    return UpstreamImgSegmentation(image_path, max_depth=max_depth,
        var_thresh=var_thresh, diff_thresh=diff_thresh,
        diff_portion=diff_portion, window_size=window_size).to_json_tree()
