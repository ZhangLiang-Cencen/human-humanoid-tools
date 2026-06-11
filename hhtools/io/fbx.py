"""FBX importer with auto-detected backends.

Why this module is shaped like a backend dispatcher rather than a single implementation:

* Autodesk's FBX file format is closed (despite the name) and the only fully-spec-compliant
  reader is Autodesk's own C++ SDK with proprietary Python bindings.  We can't ship that
  by default and most users don't have it pre-installed.

* The open-source ``libassimp`` (via ``pyassimp`` or ``assimp_py``) parses many FBX files
  but a) needs a system shared library that's often missing in minimal containers, and
  b) the lightweight ``assimp_py`` Python binding (``pip install assimp_py``) doesn't
  expose bones / animations at all — only mesh geometry.  ``pyassimp`` *does* expose
  bones/anims but requires libassimp-dev.

* ``FBX2glTF`` (Facebook's CLI tool) is, in practice, the most reliable open-source path:
  it shells out to a self-contained binary, converts to glTF 2.0, and we then route through
  the already-tested :func:`hhtools.io.glb.load_glb`.  This is the path most likely to
  succeed in unattended environments.

So this module probes the available backends *in order* and uses the first that works:

    1. ``FBX2glTF`` CLI on PATH      → convert to a temporary ``.glb``, then load_glb()
    2. Autodesk FBX SDK Python bindings → native (placeholder, raises until we wire it up)
    3. ``pyassimp`` / ``assimp_py``   → native (placeholder, raises until we wire it up)

Failed backends accumulate diagnostics, and if none succeed we raise a single
``NotImplementedError`` with three actionable workarounds — no need for the user to dig
through tracebacks.

When (1) succeeds, the returned :class:`Motion` carries
``meta["fbx_backend"] = "fbx2gltf"`` so downstream code can tell it came through a
GLB intermediate (useful when debugging unit / coordinate issues).

This file is *not* a port of any third-party FBX importer — it deliberately keeps the
native-decode work as future stubs, because the conversion path covers most real-world
needs without taking on a heavyweight C++ dependency now.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from hhtools.core.motion import Motion


class _BackendUnavailableError(Exception):
    """Internal signal that a particular backend can't run on the current machine."""


def load_fbx(
    path: str | Path,
    *,
    target_fps: float = 60.0,
    target_up_axis: str = "Z",
    animation_index: int = 0,
    joint_names: list[str] | None = None,
    **glb_kwargs: Any,
) -> Motion:
    """Load a ``.fbx`` file by trying each available backend in order.

    Args:
        path: Path to the file on disk.
        target_fps: Uniform sample rate of the resulting motion (passed to whichever
            backend ultimately produces the motion).
        target_up_axis: ``"Z"`` (default, hhtools internal convention) or ``"Y"``.
        animation_index: Which animation clip to load when the FBX contains multiple.
        joint_names: Optional whitelist of node names to keep — same semantics as
            :func:`hhtools.io.glb.load_glb`.  Only honoured by backends that route
            through the glTF intermediate, since native FBX backends pick joints from
            ``Skin.GetCluster``.

    Raises:
        FileNotFoundError: when ``path`` doesn't exist.
        NotImplementedError: when no backend is available, with a multi-line message
            describing the three supported install paths.

    Returns:
        A :class:`Motion`.  ``meta["fbx_backend"]`` records which backend won.

    Note:
        FBX is treated as **skeleton-only** on purpose: most animation FBX files
        in the wild (mocap captures, Cranberry, the ai4animationpy demos that
        pair ``*.fbx`` with a separate ``Model.glb``) carry an armature + anim
        curves but no skinned mesh.  The bpy → glTF intermediate would then
        produce a meshless GLB, and every attempt to read its (non-existent)
        mesh was brittle for users.  The reliable UX is: load skeleton here,
        render capsules/bones, and let GLB / SMPL paths handle mesh separately
        when the user actually has one.  We override any ``with_mesh=True``
        passed by the viewer so this is enforced at the backend boundary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FBX file not found: {path}")

    forwarded = dict(glb_kwargs)
    mesh_request_suppressed = bool(forwarded.pop("with_mesh", False))
    # Progress callback is viewer-specific — extract it so ``load_glb(**kwargs)``
    # deep in each backend doesn't see it.  We pass it explicitly by name to each
    # backend and let the ones that don't care simply ignore it.
    progress_callback = forwarded.pop("progress_callback", None)
    kwargs: dict[str, Any] = {
        "target_fps": target_fps,
        "target_up_axis": target_up_axis,
        "animation_index": animation_index,
        "joint_names": joint_names,
        "with_mesh": False,  # FBX path is skeleton-only — see the note above.
        **forwarded,
    }

    # Order matters: cheapest to most intrusive.  ``fbx2gltf`` shells out to a
    # self-contained binary (~50 ms); ``bpy`` runs Blender in-process (0.5–2 s first
    # call, then cached); ``blender_cli`` spawns headless Blender (2–5 s per clip but
    # works even when the Python module isn't available for our interpreter version).
    # The two Blender paths are listed separately because most users have *one* of
    # them but not both, and knowing which one was tried makes the diagnostic output
    # actionable.
    backends = [
        ("fbx2gltf", _try_fbx2gltf),
        ("bpy", _try_bpy),
        ("blender_cli", _try_blender_cli),
        ("autodesk_sdk", _try_autodesk_sdk),
        ("pyassimp", _try_pyassimp),
    ]

    diagnostics: list[str] = []
    for name, fn in backends:
        try:
            motion = fn(path, progress_callback=progress_callback, **kwargs)
        except _BackendUnavailableError as err:
            diagnostics.append(f"  - {name}: {err}")
            continue
        except NotImplementedError as err:
            diagnostics.append(f"  - {name} (detected but not yet wired up): {err}")
            continue
        # Record whether we silently dropped a mesh request — UI can surface this
        # as a non-fatal notice instead of silently ignoring the user's toggle.
        if mesh_request_suppressed:
            motion.meta["fbx_mesh_skipped"] = True
        return motion

    raise NotImplementedError(
        f"No FBX backend was usable for {path}.\n"
        "Tried (in order):\n"
        + "\n".join(diagnostics)
        + "\n\n"
        "Recommended workarounds (any one is sufficient):\n"
        "  1. Install Blender and re-run (recommended — Blender's FBX importer is the\n"
        "     best-maintained open-source backend; hhtools will pick it up automatically):\n"
        "     - apt install blender            (Debian/Ubuntu)\n"
        "     - snap install blender           (Ubuntu with snapd)\n"
        "     - or download from https://www.blender.org\n"
        "     Alternative (in-process, faster): `pip install bpy` — note the wheel is\n"
        "     pinned to a specific Python version (e.g. bpy 4.5 → Python 3.11).\n"
        "  2. Install the FBX2glTF CLI and put it on $PATH:\n"
        "       https://github.com/facebookincubator/FBX2glTF/releases\n"
        "     (binary names 'FBX2glTF' / 'fbx2gltf' / 'FBX2glTF-linux-x86_64' all work).\n"
        "  3. Install Autodesk FBX SDK Python bindings (proprietary; full spec coverage):\n"
        "       https://aps.autodesk.com/developer/overview/fbx-sdk\n"
        "     — the hhtools Autodesk backend will still need glue code before this\n"
        "     option becomes end-to-end (backend is currently a placeholder).\n"
        "  4. Install libassimp + pyassimp:\n"
        "       apt install libassimp-dev && pip install 'hhtools[fbx]'\n"
        "     (also a placeholder backend for now; detection → actionable diagnostics).\n"
    )


# ---------------------------------------------------------------- backend: FBX2glTF CLI


def _find_fbx2gltf_executable() -> str | None:
    """Locate the FBX2glTF CLI by trying every name it ships under across releases."""
    repo_bin = Path(__file__).resolve().parents[2] / "tools" / "bin"
    for name in ("FBX2glTF", "fbx2gltf", "FBX2glTF-linux-x86_64", "FBX2glTF-darwin-x86_64"):
        which = shutil.which(name)
        if which:
            return which
        bundled = repo_bin / name
        if bundled.is_file():
            return str(bundled)
    return None


def _try_fbx2gltf(
    path: Path, *, progress_callback: Any = None, **kwargs: Any,  # noqa: ARG001
) -> Motion:
    """Convert FBX → GLB via the ``FBX2glTF`` CLI, then dispatch to :func:`load_glb`.

    The CLI writes ``<output>_out/<output>.glb`` by default; we pin the layout with
    ``--output`` and read back from the deterministic location.  The temporary directory
    is cleaned up via the context manager regardless of success / failure.

    ``progress_callback`` is accepted for API symmetry with :func:`_try_bpy` but
    the CLI backend is fast enough (<100 ms typical) that live milestones add
    no value — we just let the synthetic time curve handle it.
    """
    cli = _find_fbx2gltf_executable()
    if cli is None:
        raise _BackendUnavailableError(
            "FBX2glTF CLI not on PATH (looked for FBX2glTF / fbx2gltf / "
            "FBX2glTF-linux-x86_64 / FBX2glTF-darwin-x86_64)."
        )

    with tempfile.TemporaryDirectory(prefix="hhtools_fbx2gltf_") as tmp:
        out_path = Path(tmp) / f"{path.stem}.glb"
        result = subprocess.run(
            [cli, "--input", str(path), "--output", str(out_path), "--binary"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            raise _BackendUnavailableError(
                f"FBX2glTF exited with code {result.returncode}: "
                f"{result.stderr.strip()[:200] or '(no stderr)'}"
            )
        # Some FBX2glTF builds append a suffix; if the exact name isn't there, find any .glb.
        if not out_path.exists():
            candidates = list(Path(tmp).rglob("*.glb"))
            if not candidates:
                raise _BackendUnavailableError(
                    f"FBX2glTF reported success but produced no .glb under {tmp}."
                )
            out_path = candidates[0]

        # Lazy import to avoid a circular dependency at module-load time.
        from hhtools.io.glb import load_glb

        motion = load_glb(out_path, **kwargs)
        motion.meta["fbx_backend"] = "fbx2gltf"
        motion.meta["fbx_source_path"] = str(path)
        return motion


# ---------------------------------------------------------------- backend: bpy in-process


# Blender Python script to convert FBX → GLB in an isolated subprocess.
# Reads ``src`` / ``dst`` from ``sys.argv`` (Blender convention: anything after ``--``).
# Must stay in sync with :func:`_try_bpy`'s ``subprocess.run`` call below.
_BPY_CONVERT_SCRIPT = """\
import sys
import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format='GLB',
    export_animations=True,
    export_force_sampling=True,
    export_skins=True,
    export_yup=True,
)
"""


# Milestones we recognise on the bpy subprocess's stdout and the percentage floor
# they correspond to on a typical load.  Kept loose on purpose: any matching
# substring is enough, so small Blender version changes in the text output don't
# break the progress-bar UX — they just cause us to miss a pin, falling back to
# the time-based asymptotic curve.
_BPY_MILESTONES: tuple[tuple[str, float, str], ...] = (
    ("FBX version", 12.0, "Reading FBX header"),
    ("io_scene_fbx", 18.0, "Parsing FBX scene"),
    ("build_hierarchy", 30.0, "Building Blender hierarchy"),
    ("Starting glTF 2.0 export", 60.0, "Exporting to glTF"),
    ("Finished glTF 2.0 export", 90.0, "Finished glTF export"),
)


def _try_bpy(
    path: Path, *, progress_callback: Any = None, **kwargs: Any,
) -> Motion:
    """Import FBX by driving ``bpy`` in a **subprocess**, then route through load_glb.

    Why subprocess and not in-process:

    Blender's FBX importer is the best-maintained open-source FBX reader we know of,
    and ``bpy`` (the pip-installable distribution) exposes it directly.  Unfortunately
    running ``bpy`` inside a long-lived host process — specifically Viser + asyncio /
    uvloop — has proven unstable: the first FBX import completes correctly, but at
    some later point ``bpy``'s teardown interferes with another thread's Python
    finalizer and segfaults the whole server.  That's fatal for the viewer because a
    single bad click kills every active session.

    Running ``bpy`` in a short-lived child process fixes this completely:

    * The child crashes are isolated — its exit code is propagated to us as a normal
      ``CalledProcessError``, the viewer stays up.
    * There's no scene-state bleed between calls (a fresh child starts with an empty
      :class:`bpy.context`), so we can drop the in-process ``read_factory_settings``
      gymnastics.
    * The TBB / OpenVDB symbol-clash workaround is simpler: ``LD_PRELOAD`` picks up
      ``bpy``'s bundled ``libtbb.so.12`` before the child imports anything, whereas
      the in-process fix had to go through ``ctypes.CDLL(RTLD_GLOBAL)`` inside the
      current Python.

    Cost: ~2–3 s per call for subprocess startup + ``bpy`` init, vs ~0.5–1 s
    in-process after warmup.  That's negligible for a UI-driven importer and well
    worth the stability.

    Notes:
        * ``bpy`` wheels are interpreter-specific (bpy 5.1 → Python 3.13;
          bpy 4.5 → 3.11).  We raise :class:`_BackendUnavailableError` with a clear
          install pointer when the module isn't importable, so the dispatcher can
          move on to :func:`_try_blender_cli`.
        * We reuse the *parent* interpreter (``sys.executable``) for the child, so
          everyone shares the same ``bpy`` install — no need for a second wheel.
    """
    # ``progress_callback`` is an optional hook used by the viewer: it accepts
    # ``(message: str | None, *, floor: float)`` and jumps the UI's progress-bar
    # floor forward to ``floor`` (0..100).
    progress_cb = progress_callback
    spec = importlib.util.find_spec("bpy")
    if spec is None or not spec.submodule_search_locations:
        raise _BackendUnavailableError(
            "bpy (Blender as a Python module) not installed.  "
            "Install with `pip install bpy` — note wheels are Python-version specific "
            "(bpy 5.1 → Python 3.13; bpy 5.0 → 3.11).  "
            "If pip install fails for your interpreter, the next backend "
            "(`blender_cli`) only needs a `blender` binary on $PATH."
        )

    bpy_lib_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"

    env = os.environ.copy()
    # Force bpy's bundled TBB to win over any system/conda libtbb that happened to
    # load first.  Without this the child raises ``undefined symbol: _ZN3tbb...`` the
    # moment something touches ``bpy.lib/libopenvdb.so.13`` (which uses TBB
    # internally).  LD_PRELOAD is process-local, so we don't pollute the parent.
    preload = [str(bpy_lib_dir / name) for name in ("libtbb.so.12", "libtbbmalloc.so.2")
               if (bpy_lib_dir / name).exists()]
    if preload:
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (":".join(preload) + (":" + existing if existing else ""))
    # Route Blender's user-config cache to a scratch dir — the default (~/.config/blender)
    # may be read-only under sandboxes / CI and spams warnings without it.
    env.setdefault("BLENDER_USER_CONFIG", tempfile.gettempdir())
    # ``PYTHONUNBUFFERED=1`` is essential: without it Python / libc buffer the
    # subprocess' stdout in 4 KB chunks, which on a short (<1 KB) bpy run means
    # we see *nothing* until the child exits — defeating the whole point of
    # streaming for live progress.
    env.setdefault("PYTHONUNBUFFERED", "1")

    with tempfile.TemporaryDirectory(prefix="hhtools_bpy_") as tmp:
        out_glb = Path(tmp) / f"{path.stem}.glb"
        # Stream stdout live so we can pin real bpy milestones onto the progress
        # bar.  This matters specifically for the viewer case: the caller on the
        # other side of ``progress_cb`` is waiting on a WebSocket frame and will
        # show a static 0% unless we push stage updates as they happen.  Using
        # ``Popen`` + a reader thread keeps us generic (no TTY magic) and avoids
        # the giant buffered dump we'd get from ``subprocess.run(capture_output=True)``.
        proc = subprocess.Popen(
            # ``-B`` just skips .pyc writes so we don't pollute the user's Python
            # cache with bpy-only intermediates.  We must *not* pass ``-S``: it
            # disables site.py, which is what puts the venv's site-packages (and
            # therefore ``bpy``) on sys.path.
            [
                sys.executable, "-B",
                "-c", _BPY_CONVERT_SCRIPT,
                "--", str(path), str(out_glb),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # bpy prints errors to stdout; merge streams.
            text=True,
            bufsize=1,                  # line-buffered
        )

        captured: list[str] = []
        startup_timeout = 30.0
        overall_timeout = 600.0
        try:
            assert proc.stdout is not None
            import threading, queue
            line_q: queue.Queue[str | None] = queue.Queue()

            def _reader() -> None:
                assert proc.stdout is not None
                try:
                    for ln in proc.stdout:
                        line_q.put(ln)
                finally:
                    line_q.put(None)

            reader_t = threading.Thread(target=_reader, daemon=True)
            reader_t.start()

            first_line_received = False
            t0 = time.monotonic()
            while True:
                wait = startup_timeout if not first_line_received else 10.0
                try:
                    item = line_q.get(timeout=wait)
                except queue.Empty:
                    if not first_line_received:
                        proc.kill()
                        proc.wait(timeout=5)
                        raise _BackendUnavailableError(
                            f"bpy subprocess produced no output within {startup_timeout:.0f}s "
                            f"(likely hung on import); skipping bpy backend."
                        )
                    if time.monotonic() - t0 > overall_timeout:
                        proc.kill()
                        proc.wait(timeout=5)
                        raise _BackendUnavailableError(
                            f"bpy subprocess timed out after {overall_timeout:.0f}s on {path.name}"
                        )
                    continue
                if item is None:
                    break
                captured.append(item)
                first_line_received = True
                _maybe_pin_milestone(progress_cb, item)
                if time.monotonic() - t0 > overall_timeout:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise _BackendUnavailableError(
                        f"bpy subprocess timed out after {overall_timeout:.0f}s on {path.name}"
                    )

            rc = proc.wait(timeout=10)
        except _BackendUnavailableError:
            raise
        except Exception:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            raise

        if rc != 0:
            tail = "".join(captured).strip()[-800:] or "(no output)"
            raise _BackendUnavailableError(
                f"bpy subprocess exited with code {rc}: {tail}"
            )

        if not out_glb.exists():
            raise _BackendUnavailableError(
                f"bpy subprocess finished but {out_glb.name} was not produced; "
                "check the FBX file for unsupported features."
            )

        if progress_cb is not None:
            # Signal the hand-off from "bpy wrote GLB" → "now parsing the GLB in
            # Python".  The viewer's ``_load_path`` already pins 95% on the final
            # "Building scene" step, so 92% here just makes the transition smooth.
            try:
                progress_cb("Parsing glTF intermediate", floor=92.0)
            except Exception:
                pass

        from hhtools.io.glb import load_glb  # lazy to avoid circular import

        motion = load_glb(out_glb, **kwargs)
        motion.meta["fbx_backend"] = "bpy"
        motion.meta["fbx_source_path"] = str(path)
        return motion


def _maybe_pin_milestone(progress_cb, line: str) -> None:  # type: ignore[no-untyped-def]
    """Dispatch a stdout line to ``progress_cb`` if it matches a known milestone.

    Safe to call with ``progress_cb=None`` (no-op) and catches any exception the
    callback raises — the subprocess reader thread can't be allowed to die on a
    UI hiccup, it still has to drain the pipe until bpy exits.
    """
    if progress_cb is None or not line:
        return
    for needle, floor, message in _BPY_MILESTONES:
        if needle in line:
            try:
                progress_cb(message, floor=floor)
            except Exception:
                pass
            return


# ---------------------------------------------------------------- backend: blender CLI


def _try_blender_cli(
    path: Path, *, progress_callback: Any = None, **kwargs: Any,  # noqa: ARG001
) -> Motion:
    """Spawn ``blender --background`` to convert FBX → GLB, then route through load_glb.

    This is the fallback when ``bpy`` isn't importable (common on unsupported Python
    versions like 3.13).  It requires only a ``blender`` executable on ``$PATH``; the
    binary handles all interpreter-version mismatches internally because Blender
    ships its own Python.

    Notes:
        * We pass the conversion logic via ``--python-expr`` instead of a tempfile
          script so there's nothing to clean up on crashes.  The args after ``--`` are
          forwarded to the embedded Python (Blender convention) so it can read the
          input / output paths without env-var tricks.
        * A 10-minute wall timeout catches cases where Blender hangs waiting for
          interactive input (shouldn't happen in ``--background`` but has been
          observed on misconfigured CI).
    """
    cli = shutil.which("blender")
    if cli is None:
        raise _BackendUnavailableError(
            "`blender` not on $PATH (install via `apt install blender` / "
            "`snap install blender` / download from https://www.blender.org)."
        )

    with tempfile.TemporaryDirectory(prefix="hhtools_blendercli_") as tmp:
        out_glb = Path(tmp) / f"{path.stem}.glb"
        script = (
            "import bpy, sys\n"
            "argv = sys.argv[sys.argv.index('--') + 1:]\n"
            "src, dst = argv[0], argv[1]\n"
            "for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)\n"
            "bpy.ops.import_scene.fbx(filepath=src)\n"
            "bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', "
            "export_animations=True, export_force_sampling=True, "
            "export_skins=True, export_yup=True)\n"
        )
        result = subprocess.run(
            [
                cli, "--background", "--factory-startup",
                "--python-expr", script,
                "--", str(path), str(out_glb),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0 or not out_glb.exists():
            # Blender tends to print the real error to stdout, not stderr.
            tail = (result.stderr.strip() or result.stdout.strip() or "(no output)")[-400:]
            raise _BackendUnavailableError(
                f"blender CLI exited with code {result.returncode}: {tail}"
            )

        from hhtools.io.glb import load_glb

        motion = load_glb(out_glb, **kwargs)
        motion.meta["fbx_backend"] = "blender_cli"
        motion.meta["fbx_source_path"] = str(path)
        return motion


# ---------------------------------------------------------------- backend: Autodesk SDK


def _try_autodesk_sdk(
    path: Path, *, progress_callback: Any = None, **kwargs: Any,  # noqa: ARG001
) -> Motion:
    """Autodesk FBX SDK Python bindings backend (planned).

    Currently raises ``NotImplementedError`` once the SDK *is* detected, so the dispatcher
    can include "SDK present but glue code missing" in its diagnostics — that's much
    more actionable than a silent skip when a user has gone through the trouble of
    installing the SDK.
    """
    try:
        import fbx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise _BackendUnavailableError(
            "Autodesk FBX SDK Python bindings (`import fbx`) not installed."
        ) from err
    raise NotImplementedError(
        "Autodesk FBX SDK detected, but native backend not wired up yet "
        "(planned for a future hhtools release)."
    )


# ---------------------------------------------------------------- backend: pyassimp


def _try_pyassimp(
    path: Path, *, progress_callback: Any = None, **kwargs: Any,  # noqa: ARG001
) -> Motion:
    """libassimp / pyassimp backend (planned).

    We probe ``pyassimp`` rather than ``assimp_py`` because the latter (the lightweight
    binding installable via pip without system packages) does *not* expose bones or
    animations — only mesh geometry, which is useless for skeletal motion import.
    Once ``pyassimp`` is detected we still raise ``NotImplementedError`` until the
    skeleton-extraction code is written.
    """
    try:
        import pyassimp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise _BackendUnavailableError(
            "pyassimp not installed (system libassimp + `pip install hhtools[fbx]`)."
        ) from err
    raise NotImplementedError(
        "pyassimp detected, but native FBX backend not wired up yet "
        "(planned for a future hhtools release)."
    )


__all__ = ["load_fbx"]
