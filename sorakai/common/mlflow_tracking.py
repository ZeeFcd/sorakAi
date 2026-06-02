"""Optional MLflow logging for MLOps demos.

No-op when ``Settings.mlflow_tracking_uri`` is unset. Reads the URI from
:class:`sorakai.common.config.Settings` rather than ``os.getenv`` directly,
so the rest of the code never reaches into the environment (DIP).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sorakai.common.config import get_settings
from sorakai.core.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def mlflow_run(experiment_name: str, run_name: str | None = None) -> Iterator[Any]:
    uri = get_settings().mlflow_tracking_uri
    if not uri:
        yield None
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as exc:
        # MLflow is optional; a failed import / network blip must never break a request.
        # Wave 8 of the overhaul plan replaces this with a structured callback handler.
        logger.warning("MLflow run skipped: %s", exc)
        yield None


def log_params_metrics(params: dict[str, Any], metrics: dict[str, float]) -> None:
    if not get_settings().mlflow_tracking_uri:
        return
    try:
        import mlflow

        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    except Exception as exc:
        # Same justification as above.
        logger.warning("MLflow log failed: %s", exc)
