"""WAV transport for the LPI preview GUI: play / pause / stop / seek."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

TimeFn = Callable[[], float]


def find_pyaudio_pulse_device_index(pyaudio_instance) -> Optional[int]:
    """Match audio-player shared mode: route through the Pulse/PipeWire device."""
    for index in range(pyaudio_instance.get_device_count()):
        info = pyaudio_instance.get_device_info_by_index(index)
        if info.get('maxOutputChannels', 0) < 1:
            continue
        if str(info.get('name', '')).lower() == 'pulse':
            return int(index)
    return None


def resolve_sounddevice_output_device(sample_rate: int, channels: int):
    """Pick an output device that can reach the lathe user's Pulse session."""
    import sounddevice as sd

    candidates = []
    try:
        devices = sd.query_devices()
    except Exception:
        devices = []
    if isinstance(devices, dict):
        devices = [devices]
    for info in devices:
        if int(info.get('max_output_channels', 0)) < channels:
            continue
        name = str(info.get('name', '')).lower()
        if name == 'pulse' or 'pulse' in name:
            candidates.append(info['index'])
    for fallback in ('default', 'pulse'):
        if fallback not in candidates:
            candidates.append(fallback)
    candidates.append(None)

    for device in candidates:
        try:
            sd.check_output_settings(
                device=device,
                samplerate=int(sample_rate),
                channels=int(channels),
                dtype='float32',
            )
            return device
        except Exception:
            continue
    return None


def fill_output_block(source: np.ndarray, start_frame: int, outdata: np.ndarray) -> tuple[int, bool]:
    """Copy source[start_frame:] into outdata. Returns (new_frame, finished)."""
    source = np.asarray(source)
    frames = int(outdata.shape[0])
    start_frame = int(max(0, start_frame))
    available = int(source.shape[0]) - start_frame
    if available <= 0:
        outdata.fill(0)
        return int(source.shape[0]), True
    n = min(frames, available)
    outdata[:n] = source[start_frame:start_frame + n]
    if n < frames:
        outdata[n:] = 0
    new_frame = start_frame + n
    return new_frame, new_frame >= int(source.shape[0])


class NullAudioOutput:
    """Silent stand-in so playhead tests (and headless runs) need no sound device."""

    available = True

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        return None

    def stop(self) -> None:
        return None


class SoundDeviceOutput:
    available = True

    def __init__(self):
        self._stream = None
        self._data = None
        self._frame = 0
        self._lock = threading.Lock()

    @property
    def current_frame(self) -> int:
        with self._lock:
            return int(self._frame)

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        import sounddevice as sd
        self.stop()
        data = np.asarray(samples)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = np.ascontiguousarray(data, dtype=np.float32)
        start_frame = int(max(0, min(start_frame, data.shape[0])))
        with self._lock:
            self._data = data
            self._frame = start_frame
        channels = int(data.shape[1])
        device = resolve_sounddevice_output_device(int(sample_rate), channels)

        def callback(outdata, frames, _time_info, _status):
            with self._lock:
                source = self._data
                frame = self._frame
                if source is None:
                    outdata.fill(0)
                    raise sd.CallbackStop
                new_frame, finished = fill_output_block(source, frame, outdata)
                self._frame = new_frame
            if finished:
                raise sd.CallbackStop

        self._stream = sd.OutputStream(
            samplerate=int(sample_rate),
            channels=channels,
            dtype='float32',
            blocksize=4096,
            latency='high',
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception:
                pass
        try:
            stream.close()
        except Exception:
            pass


class DisabledAudioOutput:
    available = False

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        raise RuntimeError('No audio output is available')

    def stop(self) -> None:
        return None


def create_audio_output():
    try:
        import sounddevice as sd  # noqa: F401
        return SoundDeviceOutput()
    except Exception:
        try:
            import pyaudio  # noqa: F401
            return PyAudioOutput()
        except Exception:
            return DisabledAudioOutput()


class PyAudioOutput:
    """Blocking-write fallback used on the lathe laptop if sounddevice is missing."""

    available = True

    def __init__(self):
        self._thread = None
        self._stop = False
        self._pa = None
        self._stream = None

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        import threading
        import pyaudio

        self.stop()
        data = np.asarray(samples)
        start_frame = int(max(0, min(start_frame, data.shape[0])))
        chunk = data[start_frame:]
        if chunk.size == 0:
            return
        pcm = np.clip(chunk, -1.0, 1.0)
        if pcm.ndim == 1:
            pcm = pcm.reshape(-1, 1)
        interleaved = (pcm * 32767.0).astype(np.int16).tobytes()
        channels = int(pcm.shape[1])
        self._stop = False

        def _run():
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=int(sample_rate),
                output=True,
                frames_per_buffer=4096,
            )
            self._pa = pa
            self._stream = stream
            frame_bytes = 4096 * channels * 2
            offset = 0
            try:
                while not self._stop and offset < len(interleaved):
                    stream.write(interleaved[offset:offset + frame_bytes])
                    offset += frame_bytes
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                try:
                    pa.terminate()
                except Exception:
                    pass
                self._stream = None
                self._pa = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._thread = None


class Playhead:
    def __init__(
        self,
        time_fn: TimeFn = time.monotonic,
        output: Optional[object] = None,
    ):
        self._time_fn = time_fn
        self.output = output if output is not None else NullAudioOutput()
        self.samples: Optional[np.ndarray] = None
        self.sample_rate = 1
        self.duration_s = 0.0
        self._state = 'stopped'
        self._origin_s = 0.0
        self._play_started_at = 0.0

    @property
    def is_playing(self) -> bool:
        self.position_s
        return self._state == 'playing'

    @property
    def position_s(self) -> float:
        if self._state == 'playing':
            current_frame = getattr(self.output, 'current_frame', None)
            if current_frame is not None:
                elapsed = float(current_frame) / float(self.sample_rate)
            else:
                elapsed = self._origin_s + (self._time_fn() - self._play_started_at)
            if elapsed >= self.duration_s:
                self._origin_s = self.duration_s
                self._state = 'paused'
                self._stop_output()
                return self.duration_s
            return max(0.0, elapsed)
        return max(0.0, min(self._origin_s, self.duration_s))

    def load(self, samples, sample_rate: int) -> None:
        self.stop()
        data = np.asarray(samples)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.samples = data
        self.sample_rate = max(1, int(sample_rate))
        self.duration_s = float(data.shape[0]) / float(self.sample_rate)

    def play(self) -> None:
        if self.samples is None or self.duration_s <= 0:
            return
        if self._state == 'playing':
            return
        if self._origin_s >= self.duration_s:
            self._origin_s = 0.0
        self._play_started_at = self._time_fn()
        self._state = 'playing'
        self._start_output()

    def pause(self) -> None:
        if self._state != 'playing':
            self.position_s
            return
        self._origin_s = self.position_s
        self._state = 'paused'
        self._stop_output()

    def stop(self) -> None:
        self._stop_output()
        self._state = 'stopped'
        self._origin_s = 0.0

    def seek(self, time_s: float) -> None:
        playing = self._state == 'playing'
        if playing:
            self._stop_output()
        t = 0.0 if time_s is None else float(time_s)
        self._origin_s = min(max(t, 0.0), self.duration_s)
        if playing and self._origin_s < self.duration_s:
            self._play_started_at = self._time_fn()
            self._state = 'playing'
            self._start_output()
        elif playing:
            self._state = 'paused'

    def _frame_at(self, time_s: float) -> int:
        if self.samples is None:
            return 0
        frame = int(round(time_s * self.sample_rate))
        return int(min(max(frame, 0), self.samples.shape[0]))

    def _start_output(self) -> None:
        if self.samples is None:
            return
        if not getattr(self.output, 'available', True):
            raise RuntimeError('No audio output is available')
        self.output.play(self.samples, self.sample_rate, self._frame_at(self._origin_s))

    def _stop_output(self) -> None:
        try:
            self.output.stop()
        except Exception:
            pass
