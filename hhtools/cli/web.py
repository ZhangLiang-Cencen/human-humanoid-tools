"""``hhtools web`` — launch the HTML / three.js web UI (FastAPI backend).

This is the modern replacement for the Viser viewer (``hhtools ui``).  The
browser does all 3D rendering; the backend re-uses the hhtools pipeline.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Launch the HTML web UI (Apple-styled three.js front-end).")


@app.callback(invoke_without_command=True)
def launch(
    ctx: typer.Context,
    source: Path = typer.Option(
        Path("assets/motions"),
        "--source",
        "-s",
        help="Raw-dataset root scanned recursively for the motion library.",
    ),
    save_dir: Path = typer.Option(
        Path("assets/save_npz"),
        "--save-dir",
        help="Viser-style persisted NPZ cache (web exports download via the browser).",
    ),
    cache: Path | None = typer.Option(
        None, "--cache", help="Per-session NPZ cache dir (defaults to a tempdir)."
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8009, "--port"),
) -> None:
    """Start the web UI on ``host:port`` and open a browser."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        from hhtools.web.server import run_web
    except ImportError as exc:
        typer.echo(
            "The web UI requires the optional extras. Install them with:\n"
            "    uv sync --extra web --extra retarget\n"
            "Retarget (Newton IK) also needs the NVIDIA ``newton`` package per upstream docs.\n"
            "    (or: pip install 'hhtools[web,retarget]')"
        )
        raise typer.Exit(code=1) from exc

    run_web(
        source_root=source,
        save_dir=save_dir,
        cache_dir=cache,
        host=host,
        port=port,
    )


__all__ = ["app"]
