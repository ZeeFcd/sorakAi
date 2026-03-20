"""Optional MLflow logging for MLOps demos (no-op when MLFLOW_TRACKING_URI is unset)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.mlflow")


@contextmanager
def mlflow_run(experiment_name: str, run_name: str | None = None) -> Iterator[Any]:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        yield None
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow run skipped: %s", e)
        yield None


def log_params_metrics(params: dict[str, Any], metrics: dict[str, float]) -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow

        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow log failed: %s", e)
