import asyncio
import json
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
import threading

import numpy as np
import sounddevice as sd
import websockets

from PySide6 import QtCore, QtGui, QtWidgets


BACKEND_WS = "ws://localhost:8000/ws/pitch"
DEFAULT_PRESET = "guitar_standard"
DEFAULT_ALGO = "yin"  # yin|acf
DEFAULT_SMOOTH = "ema"  # none|ema|median
DEFAULT_A4 = 440
DEFAULT_VAD_RMS = 0.01  # ~-40 dBFS gate for silence suppression
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048


@dataclass
class PitchState:
    note: str = ""
    cents_off: float = float("nan")
    target_note: str = ""
    target_freq: float = float("nan")
    cents_to_target: float = float("nan")
    f0_hz: float = 0.0
    time_s: float = 0.0


class TunerGauge(QtWidgets.QWidget):
    valueChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.cents = float("nan")  # -50..+50 good range
        self.note = ""
        self.status_text = ""
        self.tooltip_text = ""
        self._ema_cents = float("nan")
        self._ema_alpha = 0.25
        # Hysteresis + debounce for status stability
        self._status: str = ""
        self._pending_status: Optional[str] = None
        self._pending_since_s: float = 0.0
        self._debounce_s: float = 0.15  # 150 ms to confirm status change
        self._hyst_in_cents: float = 3.0
        self._hyst_out_cents: float = 6.0
        self.setMinimumSize(560, 280)

    def setReadingState(self, state: PitchState):
        self.note = state.note or state.target_note or ""
        raw_cents = state.cents_off if not math.isnan(state.cents_off) else state.cents_to_target
        # EMA smoothing on needle
        if isinstance(raw_cents, float) and not math.isnan(raw_cents):
            if math.isnan(self._ema_cents):
                self._ema_cents = raw_cents
            else:
                self._ema_cents = self._ema_alpha * raw_cents + (1 - self._ema_alpha) * self._ema_cents
        else:
            self._ema_cents = float("nan")
        self.cents = self._ema_cents
        # status with hysteresis + debounce
        self.status_text = self._update_status(raw_cents)
        # tooltip
        if state.f0_hz > 0 and state.target_note:
            self.tooltip_text = f"f0 ≈ {state.f0_hz:.2f} Hz, target {state.target_note} ({state.target_freq:.2f} Hz), offset {raw_cents:+.1f}c"
        else:
            self.tooltip_text = ""
        self.setToolTip(self.tooltip_text)
        self.update()

    def _update_status(self, raw_cents: float) -> str:
        import time as _t
        now = _t.monotonic()
        if not isinstance(raw_cents, float) or math.isnan(raw_cents):
            self._status = ""
            self._pending_status = None
            return self._status
        # Determine candidate with hysteresis around center
        if self._status == "In tune":
            # Stay in tune until we exit wider band
            if abs(raw_cents) > self._hyst_out_cents:
                candidate = "Tune up" if raw_cents > 0 else "Tune down"
            else:
                candidate = "In tune"
        else:
            # Enter in-tune only when inside tighter band
            if abs(raw_cents) < self._hyst_in_cents:
                candidate = "In tune"
            else:
                candidate = ("Tune up" if raw_cents > 0 else "Tune down")
        if candidate != self._status:
            if self._pending_status != candidate:
                self._pending_status = candidate
                self._pending_since_s = now
            else:
                if (now - self._pending_since_s) >= self._debounce_s:
                    self._status = candidate
                    self._pending_status = None
        else:
            self._pending_status = None
        return self._status

    def setSmoothingMode(self, mode: str):
        mode_l = (mode or "").lower()
        if mode_l == "none":
            self._ema_alpha = 1.0  # effectively no smoothing
        elif mode_l == "median":
            self._ema_alpha = 0.2  # keep EMA mild; backend median handles most smoothing
        else:  # ema
            self._ema_alpha = 0.25

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            rect = self.rect()

            # Gauge arc
            cx, cy = rect.center().x(), rect.bottom() - 20
            radius = min(rect.width() // 2 - 40, 160)
            start_angle = 200  # degrees
            end_angle = -20

            # Background tick marks every 5 cents from -50..+50
            r_outer = radius
            r_inner_major = radius - 32
            r_inner_minor = radius - 24
            for cents in range(-50, 55, 5):
                t = (cents + 50.0) / 100.0
                ang = math.radians(start_angle + (end_angle - start_angle) * t)
                inner = r_inner_major if cents % 10 == 0 else r_inner_minor
                pen = QtGui.QPen(QtGui.QColor(80, 85, 92) if cents % 10 else QtGui.QColor(140, 145, 152))
                pen.setWidth(3 if cents == 0 else (2 if cents % 10 == 0 else 1))
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(cx + inner * math.cos(ang), cy + inner * math.sin(ang)),
                           QtCore.QPointF(cx + r_outer * math.cos(ang), cy + r_outer * math.sin(ang)))

            # Active indicator (band) with red→orange→yellow→green gradient
            if isinstance(self.cents, float) and not math.isnan(self.cents):
                clamped = max(-50.0, min(50.0, self.cents))
                t = (clamped + 50.0) / 100.0
                ang = math.radians(start_angle + (end_angle - start_angle) * t)
                r_outer = radius
                r_inner = radius - 28
                path = QtGui.QPainterPath()
                p1 = QtCore.QPointF(cx + r_inner * math.cos(ang - 0.08), cy + r_inner * math.sin(ang - 0.08))
                p2 = QtCore.QPointF(cx + r_outer * math.cos(ang - 0.08), cy + r_outer * math.sin(ang - 0.08))
                p3 = QtCore.QPointF(cx + r_outer * math.cos(ang + 0.08), cy + r_outer * math.sin(ang + 0.08))
                p4 = QtCore.QPointF(cx + r_inner * math.cos(ang + 0.08), cy + r_inner * math.sin(ang + 0.08))
                path.moveTo(p1)
                path.lineTo(p2)
                path.lineTo(p3)
                path.lineTo(p4)
                path.closeSubpath()

                grad = QtGui.QLinearGradient(p2, p3)
                grad.setColorAt(0.0, QtGui.QColor("#EF4444"))
                grad.setColorAt(0.33, QtGui.QColor("#F97316"))
                grad.setColorAt(0.66, QtGui.QColor("#F59E0B"))
                grad.setColorAt(1.0, QtGui.QColor("#22C55E"))
                p.fillPath(path, QtGui.QBrush(grad))

            # Note text
            font = QtGui.QFont("Inter", 108)
            try:
                font.setWeight(QtGui.QFont.Weight.DemiBold)
            except AttributeError:
                # Fallback for older bindings; 600 corresponds to DemiBold
                font.setWeight(600)
            font.setStyleStrategy(QtGui.QFont.PreferAntialias)
            p.setFont(font)
            p.setPen(QtGui.QPen(QtGui.QColor("#EDEFF2")))
            p.drawText(QtCore.QRectF(0, cy - radius + 20, rect.width(), 100), QtCore.Qt.AlignCenter, self.note or "")

            # Status
            small = QtGui.QFont("Inter", 18)
            p.setFont(small)
            # Color by status
            if self.status_text == "In tune":
                status_color = QtGui.QColor("#22C55E")
            elif self.status_text == "Tune up":
                status_color = QtGui.QColor("#F59E0B")
            elif self.status_text == "Tune down":
                status_color = QtGui.QColor("#F97316")
            else:
                status_color = QtGui.QColor("#A9B0BB")
            p.setPen(QtGui.QPen(status_color))
            p.drawText(QtCore.QRectF(0, cy - radius + 110, rect.width(), 40), QtCore.Qt.AlignCenter, self.status_text)
        finally:
            p.end()


class WsClientThread(QtCore.QThread):
    readingReceived = QtCore.Signal(object)

    def __init__(self, preset: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.preset = preset
        self.a4 = DEFAULT_A4
        self.algo = DEFAULT_ALGO
        self.smooth = DEFAULT_SMOOTH
        self.vad_rms = DEFAULT_VAD_RMS
        self.mode = ""  # "", "chromatic", "manual"
        self.manual_note: str = ""
        self.queue: Queue[bytes] = Queue(maxsize=10)
        self._stop_event = threading.Event()

    def push_audio(self, chunk: bytes):
        try:
            self.queue.put_nowait(chunk)
        except Exception:
            # drop if queue full
            pass

    def stop(self):
        self._stop_event.set()

    async def _sender(self, ws: websockets.WebSocketClientProtocol):
        while not self._stop_event.is_set():
            try:
                chunk = self.queue.get(timeout=0.1)
                await ws.send(chunk)
            except Empty:
                await asyncio.sleep(0.01)

    async def _receiver(self, ws: websockets.WebSocketClientProtocol):
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    try:
                        data = json.loads(msg)
                        st = PitchState(
                            note=data.get("note", ""),
                            cents_off=float(data.get("cents_off")) if data.get("cents_off") is not None else float("nan"),
                            target_note=data.get("target_note", ""),
                            target_freq=float(data.get("target_freq")) if data.get("target_freq") is not None else float("nan"),
                            cents_to_target=float(data.get("cents_to_target")) if data.get("cents_to_target") is not None else float("nan"),
                            f0_hz=float(data.get("f0_hz")) if data.get("f0_hz") is not None else 0.0,
                            time_s=float(data.get("time")) if data.get("time") is not None else 0.0,
                        )
                        self.readingReceived.emit(st)
                    except Exception:
                        pass
        except websockets.ConnectionClosed:
            return

    async def _run_async(self):
        params = {
            "preset": self.preset,
            "a4": str(self.a4),
            "algo": self.algo,
            "smooth": self.smooth,
        }
        if self.vad_rms and self.vad_rms > 0:
            params["vad_rms"] = str(self.vad_rms)
        if self.mode:
            params["mode"] = self.mode
        if self.manual_note and self.mode == "manual":
            params["manual_note"] = self.manual_note
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        uri = f"{BACKEND_WS}?{qs}"
        async with websockets.connect(uri, max_size=None) as ws:
            await asyncio.gather(self._sender(ws), self._receiver(ws))

    def run(self):
        try:
            asyncio.run(self._run_async())
        except Exception:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tuner Desktop")

        # Left: Preset + String chips
        self.presetBox = QtWidgets.QComboBox()
        self.presetBox.addItems([
            "guitar_standard",
            "violin_standard",
            "ukulele_standard",
            "viola_standard",
            "cello_standard",
            "bass_standard",
        ])
        self.presetBox.setCurrentText(DEFAULT_PRESET)
        self.stringChipsWrap = QtWidgets.QWidget()
        self.stringChipsLayout = QtWidgets.QHBoxLayout(self.stringChipsWrap)
        self.stringChipsLayout.setContentsMargins(0, 4, 0, 0)
        self.stringChipsLayout.setSpacing(6)
        self._string_buttons: List[QtWidgets.QPushButton] = []
        self._build_string_chips(["E2","A2","D3","G3","B3","E4"])  # default guitar

        leftCol = QtWidgets.QVBoxLayout()
        leftCol.addWidget(QtWidgets.QLabel("Instrument preset"))
        leftCol.addWidget(self.presetBox)
        leftCol.addWidget(self.stringChipsWrap)
        leftCol.addStretch(1)

        # Center: Tuner cluster (Note + Meter + state)
        self.gauge = TunerGauge()
        centerCol = QtWidgets.QVBoxLayout()
        centerCol.addStretch(1)
        centerCol.addWidget(self.gauge, 1)
        centerCol.addStretch(1)

        # Right: Control Bar (Start/Stop, Input, A4, Algo, Smooth, Chromatic)
        self.startBtn = QtWidgets.QPushButton("Start")
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.stopBtn.setEnabled(False)
        self.inputDevice = QtWidgets.QComboBox()
        self.inputLevel = QtWidgets.QProgressBar()
        self.inputLevel.setRange(0, 100)
        self.inputLevel.setTextVisible(False)
        self._pending_input_level = 0
        self.a4Box = QtWidgets.QComboBox()
        self.a4Box.addItems(["440","442"]) 
        self.algoBox = QtWidgets.QComboBox()
        self.algoBox.addItems(["acf","yin"]) 
        self.smoothBox = QtWidgets.QComboBox()
        self.smoothBox.addItems(["none","ema","median"]) 
        self.chromaticCheck = QtWidgets.QCheckBox("Chromatic")

        rightCol = QtWidgets.QVBoxLayout()
        rightCol.addWidget(QtWidgets.QLabel("Control"))
        rightCol.addWidget(self.startBtn)
        rightCol.addWidget(self.stopBtn)
        rightCol.addSpacing(8)
        rightCol.addWidget(QtWidgets.QLabel("Input device"))
        rightCol.addWidget(self.inputDevice)
        rightCol.addWidget(self.inputLevel)
        rightCol.addSpacing(8)
        rightCol.addWidget(QtWidgets.QLabel("A4"))
        rightCol.addWidget(self.a4Box)
        rightCol.addWidget(QtWidgets.QLabel("Algorithm"))
        rightCol.addWidget(self.algoBox)
        rightCol.addWidget(QtWidgets.QLabel("Smoothing"))
        rightCol.addWidget(self.smoothBox)
        rightCol.addWidget(self.chromaticCheck)
        rightCol.addStretch(1)

        # Main layout: left | center | right
        main = QtWidgets.QHBoxLayout()
        leftWrap = QtWidgets.QWidget(); leftWrap.setLayout(leftCol)
        centerWrap = QtWidgets.QWidget(); centerWrap.setLayout(centerCol)
        rightWrap = QtWidgets.QWidget(); rightWrap.setLayout(rightCol)
        main.addWidget(leftWrap, 0)
        main.addWidget(centerWrap, 1)
        main.addWidget(rightWrap, 0)

        w = QtWidgets.QWidget()
        w.setLayout(main)
        self.setCentralWidget(w)

        self._stream: Optional[sd.InputStream] = None
        self._buffer = bytearray()
        self._ws_thread: Optional[WsClientThread] = None

        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.presetBox.currentTextChanged.connect(self.on_preset_changed)
        self.a4Box.currentTextChanged.connect(self.on_a4_changed)
        self.algoBox.currentTextChanged.connect(self.on_algo_changed)
        self.smoothBox.currentTextChanged.connect(self.on_smooth_changed)
        self.chromaticCheck.stateChanged.connect(self.on_chromatic_changed)

        # UI thread timer to apply input level safely
        self._level_timer = QtCore.QTimer(self)
        self._level_timer.setInterval(50)
        self._level_timer.timeout.connect(lambda: self.inputLevel.setValue(self._pending_input_level))
        self._level_timer.start()

        # Apply dark theme + QSS
        self.apply_theme()

    def _build_string_chips(self, strings: List[str]):
        # Clear
        for b in self._string_buttons:
            b.setParent(None)
        self._string_buttons.clear()
        for s in strings:
            btn = QtWidgets.QPushButton(s)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, name=s: self.on_string_selected(checked, name))
            self.stringChipsLayout.addWidget(btn)
            self._string_buttons.append(btn)

    def apply_theme(self):
        QtWidgets.QApplication.setStyle("Fusion")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#121417"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#EDEFF2"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#1C1F24"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(38, 41, 45))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(230, 230, 230))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(30, 33, 36))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#EDEFF2"))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#1C1F24"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#EDEFF2"))
        palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#22C55E"))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QLabel { color: #EDEFF2; }
            QComboBox { background:#1C1F24; padding:6px 10px; border:1px solid #2A2F36; border-radius:8px; color:#EDEFF2; }
            QProgressBar { background:#1C1F24; border:1px solid #2A2F36; border-radius:6px; height:10px; }
            QProgressBar::chunk { background-color:#22C55E; }
            QPushButton { background:#1C1F24; padding:8px 14px; border:1px solid #2A2F36; border-radius:10px; color:#EDEFF2; }
            QPushButton:hover { background:#22262C; }
            QPushButton:pressed { background:#171A1E; }
            QPushButton:disabled { background:#171A1E; color:#888; }
            QCheckBox { color:#EDEFF2; }
            """
        )

    def audio_callback(self, indata, frames, time, status):  # noqa: ARG002
        if self._ws_thread is None:
            return
        # indata is float32 in [-1, 1]
        self._buffer += indata.astype(np.float32).tobytes()
        # update volume level (UI thread safe via signal could be added; quick approx here)
        try:
            rms = float(np.sqrt(np.mean(indata[:, 0] ** 2) + 1e-12)) if indata.ndim > 1 else float(np.sqrt(np.mean(indata ** 2) + 1e-12))
            self._pending_input_level = max(0, min(100, int(rms * 300)))
        except Exception:
            self._pending_input_level = 0
        # Send in chunks ~100ms for smoother UI
        target_bytes = int(0.1 * SAMPLE_RATE) * 4  # float32 bytes
        if len(self._buffer) >= target_bytes:
            chunk = bytes(self._buffer[:target_bytes])
            del self._buffer[:target_bytes]
            self._ws_thread.push_audio(chunk)

    def start_streaming(self):
        self._ws_thread = WsClientThread(self.presetBox.currentText())
        self._ws_thread.a4 = int(self.a4Box.currentText())
        self._ws_thread.algo = self.algoBox.currentText()
        self._ws_thread.smooth = self.smoothBox.currentText()
        self._ws_thread.mode = "chromatic" if self.chromaticCheck.isChecked() else ""
        self._ws_thread.readingReceived.connect(self.gauge.setReadingState)
        self._ws_thread.start()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self.audio_callback,
        )
        self._stream.start()

    def stop_streaming(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._ws_thread is not None:
            self._ws_thread.stop()
            self._ws_thread.wait(1000)
            self._ws_thread = None
        self._buffer.clear()

    def on_start(self):
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.start_streaming()

    def on_stop(self):
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.stop_streaming()

    # UI events
    def on_preset_changed(self, key: str):
        # update string chips by preset
        mapping = {
            "guitar_standard": ["E2","A2","D3","G3","B3","E4"],
            "violin_standard": ["G3","D4","A4","E5"],
            "ukulele_standard": ["G4","C4","E4","A4"],
            "viola_standard": ["C3","G3","D4","A4"],
            "cello_standard": ["C2","G2","D3","A3"],
            "bass_standard": ["E1","A1","D2","G2"],
        }
        self._build_string_chips(mapping.get(key, []))

    def on_a4_changed(self, text: str):
        if self._ws_thread is not None:
            self._ws_thread.a4 = int(text)
        self._restart_streaming_if_active()

    def on_algo_changed(self, text: str):
        if self._ws_thread is not None:
            self._ws_thread.algo = text
        self._restart_streaming_if_active()

    def on_smooth_changed(self, text: str):
        if self._ws_thread is not None:
            self._ws_thread.smooth = text
        self.gauge.setSmoothingMode(text)
        self._restart_streaming_if_active()

    def on_chromatic_changed(self, state: int):
        if self._ws_thread is not None:
            self._ws_thread.mode = "chromatic" if state == QtCore.Qt.Checked else ""
        self._restart_streaming_if_active()

    def on_string_selected(self, checked: bool, note_name: str):
        # Toggle manual lock per chip; single selection behavior
        if checked:
            for b in self._string_buttons:
                if b.text() != note_name:
                    b.setChecked(False)
            if self._ws_thread is not None:
                self._ws_thread.mode = "manual"
                self._ws_thread.manual_note = note_name
        else:
            if self._ws_thread is not None:
                self._ws_thread.mode = "chromatic" if self.chromaticCheck.isChecked() else ""
                self._ws_thread.manual_note = ""
        self._restart_streaming_if_active()

    def _restart_streaming_if_active(self):
        # Apply parameter changes immediately if currently running
        if self._stream is not None:
            self.stop_streaming()
            self.start_streaming()

    # Hotkeys
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key_Space:
            if self.stopBtn.isEnabled():
                self.on_stop()
            else:
                self.on_start()
            e.accept(); return
        if e.key() == QtCore.Qt.Key_C:
            self.chromaticCheck.setChecked(not self.chromaticCheck.isChecked())
            e.accept(); return
        if e.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            idx = self.presetBox.currentIndex()
            if e.key() == QtCore.Qt.Key_Up:
                idx = max(0, idx - 1)
            else:
                idx = min(self.presetBox.count() - 1, idx + 1)
            self.presetBox.setCurrentIndex(idx)
            e.accept(); return

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(720, 420)
    win.show()
    app.exec()


