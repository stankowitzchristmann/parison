from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from parison.furnace.twin import TherapyTwin
from parison.gather.windows import Foundry, Norm
from parison.lehr.ledger import revive
from parison.lehr.trainer import Glassworks
from parison.pipe.export import GlassPort, _dummy
from parison.proof.harness import evaluate


def test_end_to_end_train_eval(check_recipe, tmp_path: Path) -> None:
    works = Glassworks(check_recipe, torch.device("cpu"), tmp_path)
    works.fit(max_epochs=2)
    payload = revive(tmp_path / "best.ckpt")
    model = TherapyTwin(check_recipe.glass, check_recipe.feed)
    model.load_state_dict(payload["model"])
    val = Foundry(check_recipe.feed, check_recipe.glass, check_recipe.bench.seed, "val")
    val.attach_norm(Norm(payload["extra"]["norm"]))
    report = evaluate(model, val, torch.device("cpu"))
    assert np.isfinite(report["forecast_mae"])
    assert report["forecast_mae"] >= 0.0
    for key, value in report.items():
        assert np.isfinite(value), key


def test_onnx_parity(check_recipe, tmp_path: Path) -> None:
    import onnxruntime as ort

    twin = TherapyTwin(check_recipe.glass, check_recipe.feed).eval()
    port = GlassPort(twin).eval()
    args = _dummy(check_recipe)
    path = tmp_path / "twin.onnx"
    import inspect

    kwargs = {
        "input_names": ["waves", "ehr", "presence", "actions"],
        "output_names": ["forecast", "warning", "outcome"],
        "opset_version": 13,
        "dynamic_axes": {"waves": {1: "time"}, "ehr": {1: "time"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(port, args, str(path), **kwargs)

    with torch.no_grad():
        ref = port(*args)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {n: a.numpy() for n, a in zip(["waves", "ehr", "presence", "actions"], args)}
    got = sess.run(None, feeds)
    assert np.allclose(ref[0].numpy(), got[0], atol=1e-4)
    assert np.allclose(ref[1].numpy(), got[1], atol=1e-4)
