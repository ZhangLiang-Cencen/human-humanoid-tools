"""Timeline + play/pause controls for the Viser viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PlaybackState:
    frame: int = 0
    playing: bool = False
    speed: float = 1.0


class PlaybackPanel:
    """Timeline + play/pause backed by Viser GUI primitives.

    Same idea as ``soma-retargeter/app/visualize_rp1_lafan_motion.py``: scrub by
    **time in seconds**, adjust **speed** (including negative for reverse), optional
    **loop**.  Integer frame is derived internally for renderers — there is no separate
    frame slider.
    """

    def __init__(self, server, framerate: float = 30.0, num_frames: int = 1) -> None:  # type: ignore[no-untyped-def]
        self._server = server
        self._framerate = float(framerate)
        self._num_frames = int(num_frames)
        self._duration = self._compute_duration()
        self.state = PlaybackState()
        self._float_frame = 0.0
        self._last_real_time = time.perf_counter()
        self._slider_sync = False

        t_max = max(self._duration, 1e-6)
        t_step = max(1e-4, 1.0 / max(240.0, self._framerate * 4.0))

        with server.gui.add_folder("Playback"):
            self._time_slider = server.gui.add_slider(
                "Time (s)",
                min=0.0,
                max=t_max,
                step=t_step,
                initial_value=0.0,
            )
            self._play_button = server.gui.add_button("Play")
            self._pause_button = server.gui.add_button("Pause")
            self._loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
            self._speed_slider = server.gui.add_slider(
                "Speed", min=-2.0, max=4.0, step=0.1, initial_value=1.0
            )
            self._fps_label = server.gui.add_text("FPS", initial_value=f"{self._framerate:.2f}")
            self._fps_label.disabled = True
            self._status_label = server.gui.add_text("Status", initial_value="paused")
            self._status_label.disabled = True

        @self._time_slider.on_update
        def _on_time(_):  # type: ignore[no-untyped-def]
            if self._slider_sync:
                return
            t = float(self._time_slider.value)
            self._apply_time_seek(t)

        @self._play_button.on_click
        def _on_play(_):  # type: ignore[no-untyped-def]
            self.state.playing = True
            self._last_real_time = time.perf_counter()
            self._status_label.value = "playing"

        @self._pause_button.on_click
        def _on_pause(_):  # type: ignore[no-untyped-def]
            self.state.playing = False
            self._status_label.value = "paused"

        @self._speed_slider.on_update
        def _on_speed(_):  # type: ignore[no-untyped-def]
            self.state.speed = float(self._speed_slider.value)

    def _compute_duration(self) -> float:
        if self._num_frames <= 1 or self._framerate <= 0.0:
            return 0.0
        return float(self._num_frames - 1) / self._framerate

    def _sync_time_slider_from_frame(self) -> None:
        if self._framerate <= 0.0:
            t = 0.0
        else:
            t = float(self.state.frame) / self._framerate
        t = float(max(0.0, min(t, self._duration)))
        self._slider_sync = True
        try:
            self._time_slider.value = t
        finally:
            self._slider_sync = False

    def _apply_time_seek(self, t: float) -> None:
        t = float(max(0.0, min(t, self._duration)))
        if self._framerate <= 0.0:
            self._float_frame = 0.0
            self.state.frame = 0
        else:
            self._float_frame = t * self._framerate
            self._float_frame = float(
                max(0.0, min(self._float_frame, float(max(0, self._num_frames - 1))))
            )
            self.state.frame = int(round(self._float_frame))
            self.state.frame = int(max(0, min(self.state.frame, max(0, self._num_frames - 1))))
        self._last_real_time = time.perf_counter()

    def set_motion(self, framerate: float, num_frames: int, *, resume_playing: bool = False) -> None:
        self._framerate = float(framerate)
        self._num_frames = int(num_frames)
        self._duration = self._compute_duration()
        t_max = max(self._duration, 1e-6)
        t_step = max(1e-4, 1.0 / max(240.0, self._framerate * 4.0))
        self._time_slider.max = t_max
        self._time_slider.step = t_step
        self._slider_sync = True
        try:
            self._time_slider.value = 0.0
        finally:
            self._slider_sync = False
        self.state.frame = 0
        self._float_frame = 0.0
        self.state.playing = bool(resume_playing)
        self._fps_label.value = f"{self._framerate:.2f}"
        self._status_label.value = "playing" if resume_playing else "paused"
        self._last_real_time = time.perf_counter()

    def reconfigure(self, framerate: float, num_frames: int) -> int:
        framerate = float(framerate)
        num_frames = int(num_frames)
        if framerate == self._framerate and num_frames == self._num_frames:
            return self.state.frame

        self._framerate = framerate
        self._num_frames = num_frames
        self._duration = self._compute_duration()
        t_max = max(self._duration, 1e-6)
        t_step = max(1e-4, 1.0 / max(240.0, self._framerate * 4.0))
        self._time_slider.max = t_max
        self._time_slider.step = t_step
        if self.state.frame >= num_frames:
            self.state.frame = max(0, num_frames - 1)
            self._float_frame = float(self.state.frame)
        self._fps_label.value = f"{framerate:.2f}"
        self._sync_time_slider_from_frame()
        return self.state.frame

    def tick(self) -> int:
        if self._num_frames <= 1:
            return 0
        now = time.perf_counter()
        dt = now - self._last_real_time
        self._last_real_time = now
        if self.state.playing and abs(self.state.speed) > 1e-8:
            self._float_frame += dt * self._framerate * self.state.speed
            n = float(self._num_frames)
            if self._loop_checkbox.value:
                self._float_frame = self._float_frame % n
                if self._float_frame < 0.0:
                    self._float_frame += n
            else:
                if self._float_frame >= n:
                    self._float_frame = float(max(0, self._num_frames - 1))
                    self.state.playing = False
                    self._status_label.value = "paused"
                elif self._float_frame < 0.0:
                    self._float_frame = 0.0
                    self.state.playing = False
                    self._status_label.value = "paused"
            new_frame = int(self._float_frame)
            new_frame = int(max(0, min(new_frame, self._num_frames - 1)))
            if new_frame != self.state.frame:
                self.state.frame = new_frame
                self._sync_time_slider_from_frame()
        return self.state.frame

    def pause_at_frame_zero(self) -> None:
        self.state.playing = False
        self._float_frame = 0.0
        self.state.frame = 0
        self._slider_sync = True
        try:
            self._time_slider.value = 0.0
        finally:
            self._slider_sync = False
        self._last_real_time = time.perf_counter()
        self._status_label.value = "paused"

    def set_calibration_lock(self, locked: bool) -> None:
        """Enable or disable Play / time scrub / speed while the Robot tab calibrates.

        When *locked* is true, the user cannot start playback or change the
        timeline — the Motion tab stays on frame 0 for visual alignment
        with the static reference human.
        """
        lock = bool(locked)
        self._time_slider.disabled = lock
        self._play_button.disabled = lock
        self._pause_button.disabled = lock
        self._loop_checkbox.disabled = lock
        self._speed_slider.disabled = lock


__all__ = ["PlaybackPanel", "PlaybackState"]
