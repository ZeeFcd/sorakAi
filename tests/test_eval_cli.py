"""Wave 9 ``scripts/eval.py`` smoke tests.

Imports the CLI as a module and drives ``main`` with a stub LLM patched
in via ``run_eval`` so the test never touches Ollama or MLflow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sorakai.eval.dataset import EvalCase, EvalDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval.py"


@pytest.fixture
def cli_module() -> Iterator[Any]:
    """Load ``scripts/eval.py`` as a module without invoking ``main``."""
    spec = importlib.util.spec_from_file_location("sorakai_eval_cli", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["sorakai_eval_cli"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("sorakai_eval_cli", None)


def _tiny_dataset() -> EvalDataset:
    return EvalDataset(
        cases=(
            EvalCase(
                id="mars",
                question="What are the moons of Mars?",
                expected_substrings=("Phobos",),
                expected_doc_ids=("mars",),
            ),
        ),
        corpus=(("mars", "Mars has two natural moons, Phobos and Deimos."),),
    )


def test_cli_exits_zero_when_pass_rate_above_threshold(
    monkeypatch: pytest.MonkeyPatch, cli_module: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "load_default_corpus", _tiny_dataset)
    llm = FakeListChatModel(responses=["Mars has Phobos and Deimos."])

    async def _stubbed_run_eval(**kwargs: Any) -> Any:
        kwargs["llm"] = llm
        from sorakai.eval.runner import run_eval as real

        return await real(**kwargs)

    monkeypatch.setattr(cli_module, "run_eval", _stubbed_run_eval)

    exit_code = cli_module.main(["--target", "chain", "--min-pass-rate", "0.5"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "pass_rate" in captured.out
    assert "REGRESSION" not in captured.err


def test_cli_returns_nonzero_when_pass_rate_below_gate(
    monkeypatch: pytest.MonkeyPatch, cli_module: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "load_default_corpus", _tiny_dataset)
    llm = FakeListChatModel(responses=["totally unrelated answer"])

    async def _stubbed_run_eval(**kwargs: Any) -> Any:
        kwargs["llm"] = llm
        from sorakai.eval.runner import run_eval as real

        return await real(**kwargs)

    monkeypatch.setattr(cli_module, "run_eval", _stubbed_run_eval)

    exit_code = cli_module.main(["--target", "chain", "--min-pass-rate", "0.5"])
    assert exit_code == cli_module.EXIT_REGRESSION
    captured = capsys.readouterr()
    assert "REGRESSION" in captured.err


def test_cli_writes_json_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    cli_module: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "load_default_corpus", _tiny_dataset)
    llm = FakeListChatModel(responses=["Mars has Phobos and Deimos."])

    async def _stubbed_run_eval(**kwargs: Any) -> Any:
        kwargs["llm"] = llm
        from sorakai.eval.runner import run_eval as real

        return await real(**kwargs)

    monkeypatch.setattr(cli_module, "run_eval", _stubbed_run_eval)

    out = tmp_path / "nested" / "eval.json"
    exit_code = cli_module.main(["--target", "chain", "--min-pass-rate", "0", "--json", str(out)])
    assert exit_code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["target"] == "chain"
    assert payload["metrics"]["cases"] == 1.0
    assert payload["cases"][0]["case_id"] == "mars"


def test_cli_limit_truncates_dataset(monkeypatch: pytest.MonkeyPatch, cli_module: Any) -> None:
    ds = EvalDataset(
        cases=tuple(
            EvalCase(
                id=f"c{i}",
                question="Q",
                expected_substrings=("Phobos",),
                expected_doc_ids=("mars",),
            )
            for i in range(5)
        ),
        corpus=(("mars", "Mars has Phobos and Deimos."),),
    )
    monkeypatch.setattr(cli_module, "load_default_corpus", lambda: ds)
    llm = FakeListChatModel(responses=["Phobos found"] * 5)

    captured_kwargs: dict[str, Any] = {}

    async def _stubbed_run_eval(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        kwargs["llm"] = llm
        from sorakai.eval.runner import run_eval as real

        return await real(**kwargs)

    monkeypatch.setattr(cli_module, "run_eval", _stubbed_run_eval)

    exit_code = cli_module.main(["--target", "chain", "--limit", "2", "--min-pass-rate", "0"])
    assert exit_code == 0
    assert len(captured_kwargs["dataset"].cases) == 2


def test_cli_zero_cases_after_limit_returns_usage_exit(monkeypatch: pytest.MonkeyPatch, cli_module: Any) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_default_corpus",
        lambda: EvalDataset(cases=(), corpus=()),
    )
    exit_code = cli_module.main(["--target", "chain"])
    assert exit_code == cli_module.EXIT_USAGE
