"""``hhtools ui`` — launch the Viser + FastAPI web viewer."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Launch the Viser-based web viewer.")


@app.callback(invoke_without_command=True)
def launch(
    ctx: typer.Context,
    motion: Path | None = typer.Option(None, "--motion", help="Optional NPZ file to preload."),
    motion_dir: Path | None = typer.Option(
        None,
        "--motion-dir",
        help="(Legacy) flat directory of pre-converted NPZ files. Shown as an extra "
        "dropdown alongside the folder-indexed library.",
    ),
    source: Path = typer.Option(
        Path("assets/motions"),
        "--source",
        "-s",
        help="Raw-dataset root scanned recursively for the folder-indexed library. "
        "Intermediate grouping folders (mimic/ intermimic/ meshmimic/ ...) are "
        "transparent — only the innermost dataset directory names matter.",
    ),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help="Per-session NPZ cache directory. Defaults to a fresh tempfile.mkdtemp "
        "under /tmp that is rmtree'd on shutdown regardless of saves.",
    ),
    save_dir: Path = typer.Option(
        Path("assets/save_npz"),
        "--save-dir",
        help="Destination for NPZs the user explicitly persists via the 'Save' buttons.",
    ),
    keep_cache: bool = typer.Option(
        False,
        "--keep-cache/--drop-cache",
        help="Debug-only escape hatch: keep NPZs on disk after shutdown. "
        "Normal users should never need this.",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8008, "--port"),
    share: bool = typer.Option(False, "--share", help="Expose the Viser URL on the LAN."),
) -> None:
    """Start the interactive viewer on ``host:port``."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        from hhtools.viewer.app import run_viewer
    except ImportError as exc:
        typer.echo(
            "The viewer requires the optional extras. Install them with:\n"
            "    pip install 'hhtools[viewer]'"
        )
        raise typer.Exit(code=1) from exc

    run_viewer(
        motion=motion,
        motion_dir=motion_dir,
        source_root=source,
        cache_dir=cache,
        save_dir=save_dir,
        keep_cache=keep_cache,
        host=host,
        port=port,
        share=share,
    )


__all__ = ["app"]
