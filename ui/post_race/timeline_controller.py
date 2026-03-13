"""
Timeline controller - manages playback state and synchronized timeline scrubbing.
"""
from PyQt5 import QtCore
from typing import Optional


class TimelineController(QtCore.QObject):
    """
    Controls playback timeline for lap telemetry visualization.

    Similar to a video player - supports play, pause, seek, and emits
    signals when the current timestamp changes so all visualizations
    can update synchronously.
    """

    time_changed = QtCore.pyqtSignal(float)
    playback_started = QtCore.pyqtSignal()
    playback_paused = QtCore.pyqtSignal()
    playback_finished = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)

        self._current_time: float = 0.0
        self._duration: float = 0.0
        self._is_playing: bool = False
        self._playback_speed: float = 1.0

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer_interval = 16  # ~60 FPS

    def set_duration(self, duration: float) -> None:
        """Set total lap duration."""
        self._duration = duration
        if self._current_time > self._duration:
            self.seek(0.0)

    def seek(self, time: float) -> None:
        """Seek to a specific timestamp."""
        time = max(0.0, min(time, self._duration))
        if time != self._current_time:
            self._current_time = time
            self.time_changed.emit(self._current_time)

    def play(self) -> None:
        """Start playback from current position."""
        if not self._is_playing:
            self._is_playing = True
            self._timer.start(self._timer_interval)
            self.playback_started.emit()

    def pause(self) -> None:
        """Pause playback."""
        if self._is_playing:
            self._is_playing = False
            self._timer.stop()
            self.playback_paused.emit()

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self.pause()
        self.seek(0.0)

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        self._playback_speed = max(0.1, min(speed, 10.0))

    def _on_timer_tick(self) -> None:
        """Called on each timer tick during playback."""
        delta = (self._timer_interval / 1000.0) * self._playback_speed
        new_time = self._current_time + delta

        if new_time >= self._duration:
            self._current_time = self._duration
            self.time_changed.emit(self._current_time)
            self.pause()
            self.playback_finished.emit()
        else:
            self._current_time = new_time
            self.time_changed.emit(self._current_time)

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    def get_progress_percentage(self) -> float:
        if self._duration == 0:
            return 0.0
        return (self._current_time / self._duration) * 100.0
