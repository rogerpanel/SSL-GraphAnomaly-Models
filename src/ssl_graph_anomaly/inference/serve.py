"""FastAPI HTTP server exposing the streaming detector."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from ssl_graph_anomaly.inference.streaming import StreamingInference

_MODEL_ID = 14
_MODEL_NAME = "SSL-GraphAnomaly"


def build_app(
    checkpoint_path: str | Path,
    conformal_path: str | Path,
    device: str = "cpu",
) -> Any:
    """Construct the FastAPI app; raises ImportError if FastAPI is missing."""
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class DetectRequest(BaseModel):
        flow_features: list[float] = Field(..., min_length=1)

    detector = StreamingInference(
        checkpoint_path=str(checkpoint_path),
        conformal_path=str(conformal_path),
        device=device,
    )
    app = FastAPI(title=_MODEL_NAME, version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_id": _MODEL_ID, "model_name": _MODEL_NAME}

    @app.post("/api/v1/detect")
    def detect(req: DetectRequest) -> dict[str, Any]:
        result = detector.predict(req.flow_features)
        if isinstance(result, list):
            return {"results": result}
        return result

    return app


def serve_api(
    checkpoint_path: str | Path,
    conformal_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cpu",
) -> None:
    import uvicorn

    app = build_app(checkpoint_path, conformal_path, device=device)
    uvicorn.run(app, host=host, port=int(port))


@click.command()
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True), required=True)
@click.option("--conformal", "conformal_path", type=click.Path(exists=True), required=True)
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8000)
@click.option("--device", type=str, default="cpu")
def cli(
    checkpoint_path: str,
    conformal_path: str,
    host: str,
    port: int,
    device: str,
) -> None:
    serve_api(checkpoint_path, conformal_path, host=host, port=port, device=device)


if __name__ == "__main__":
    cli()
