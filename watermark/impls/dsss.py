"""DSSS "bed" scheme: a 64-bit payload in a spread-spectrum room-tone bed (8 kHz telephony).

The bed is +-1 pseudo-random chips at 8000 chips/s, filtered to a room-tone spectrum: that of a
reference recording (option bed_path, default $DSSS_BED_PATH), else a generic room-tone shape. The embedder adds it to the outgoing audio (the
bot's voice, or zeros), so the payload doesn't depend on the bot speaking.

Frame (repeats for as long as the bed plays, optionally followed by a gap of gap_s seconds):
    [ SYNC: sync_chips of known PN ] [ DATA: coded symbols x chips_per_symbol, each = +-data PN ] [ GAP ]
The gap carries no data (shaped filler noise, so the bed sounds continuous, or silence). It
lets a detector that finishes, or restarts at, the end of one frame catch the next frame's sync.

Payload: `payload_bits` (64 by default, 128 supported; any multiple of 8 up to 256) + CRC-16/CCITT
-> rate-1/2 K=7 convolutional code (171,133 octal; 6 tail bits) -> fixed pseudo-random interleaver
-> BPSK symbols: 2 x (payload_bits + 22) per frame (172 for 64 bits, 300 for 128).

Detector (streaming):
  1. equalize the bed shaping (inverse filter, 300-3400 Hz), normalize loudness per 10 ms
  2. find SYNC by normalized correlation -> exact sample position and polarity, any provider
  3. despread each data symbol; every few symbols re-check timing at +-1 packet (jitter buffer)
     and +-1 sample (clock drift)
  4. soft Viterbi + CRC; if a frame fails, combine soft symbols across repeated frames
     (up to 3, then re-acquire sync)
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..base import BaseScheme
from ..spi import Detection, Detector, Embedder

_FS = 8000
_PACKET = 160  # 20 ms
_MAX_PAYLOAD_BITS = 256
_CRC_BITS = 16
_K = 7  # convolutional code constraint length
_POLYS = (0o171, 0o133)


@dataclass
class DsssOptions:
    payload_bits: int = 64  # multiple of 8, up to 256 (e.g. 128); longer payload = longer frame
    level_dbfs: float = -47.0  # bed level while the bot is silent
    speech_margin_db: float | None = None  # raise the bed under speech to (speech - N dB); None = constant
    chips_per_symbol: int = 120  # spreading per coded bit (frame = sync + 2 x (payload_bits + 22) x this chips)
    sync_chips: int = 2048
    # recording whose spectrum the bed imitates (sender and receiver must agree); None = generic room tone
    bed_path: Path | None = field(default_factory=lambda: os.environ.get("DSSS_BED_PATH") or None)
    gap_s: float = 0.0  # pause after each frame; 0 = frames back to back
    gap_fill: str = "noise"  # "noise" = data-free bed (sounds continuous) or "silence"


class DsssScheme(BaseScheme):
    name = "dsss"
    Options = DsssOptions
    sample_rate = _FS

    def __init__(self, options=None):
        super().__init__(options)
        bits = self.options.payload_bits
        if bits % 8 or not 8 <= bits <= _MAX_PAYLOAD_BITS:
            raise ValueError(f"dsss payload_bits must be a multiple of 8 in 8..{_MAX_PAYLOAD_BITS}, got {bits}")
        self.payload_bits = bits
        if self.options.gap_s < 0 or self.options.gap_fill not in ("noise", "silence"):
            raise ValueError("dsss gap_s must be >= 0 and gap_fill 'noise' or 'silence'")

    def embedder(self, payload: bytes) -> Embedder:
        return _DsssEmbedder(_to_bits(self.check_payload(payload)), self.options)

    def detector(self) -> Detector:
        return _DsssDetector(self.options)

    def describe(self) -> str:
        gap = _gap_chips(self.options)
        return super().describe() + f"; frame {_frame_chips(self.options) / _FS:.2f}s" + (
            f" + gap {gap / _FS:.2f}s ({self.options.gap_fill})" if gap else "")


# --- payload coding ------------------------------------------------------------------------
def _to_bits(payload: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, np.uint8))


def _to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _crc16(bits: np.ndarray) -> np.ndarray:
    """CRC-16/CCITT (poly 0x1021, init 0xFFFF) over a bit array, as 16 bits."""
    reg = 0xFFFF
    for b in bits:
        top = (reg >> 15) & 1
        reg = (reg << 1) & 0xFFFF
        if top ^ int(b):
            reg ^= 0x1021
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], np.uint8)


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


# trellis of the convolutional code: next state and +-1 output symbols (bit 0 -> +1) per (state, bit)
_N_STATES = 1 << (_K - 1)
_NEXT = np.zeros((_N_STATES, 2), np.int64)
_OUT = np.zeros((_N_STATES, 2, 2), np.float32)
for _s in range(_N_STATES):
    for _b in (0, 1):
        _reg = (_b << (_K - 1)) | _s
        _NEXT[_s, _b] = _reg >> 1
        _OUT[_s, _b] = [1 - 2 * _parity(_reg & p) for p in _POLYS]
# each next-state has exactly two (state, bit) predecessors
_PRED_S = np.zeros((_N_STATES, 2), np.int64)
_PRED_B = np.zeros((_N_STATES, 2), np.int64)
_fill = np.zeros(_N_STATES, np.int64)
for _s in range(_N_STATES):
    for _b in (0, 1):
        _n = _NEXT[_s, _b]
        _PRED_S[_n, _fill[_n]], _PRED_B[_n, _fill[_n]] = _s, _b
        _fill[_n] += 1
del _s, _b, _reg, _n, _fill


def _conv_encode(bits: np.ndarray) -> np.ndarray:
    state, out = 0, []
    for b in list(bits) + [0] * (_K - 1):
        reg = (int(b) << (_K - 1)) | state
        out += [_parity(reg & p) for p in _POLYS]
        state = reg >> 1
    return np.array(out, np.uint8)


def _viterbi(soft: np.ndarray) -> np.ndarray:
    """Soft-decision Viterbi. soft: (2*T,) values, positive = bit 0. Tail-terminated."""
    soft = soft.reshape(-1, 2)
    T = len(soft)
    pm = np.full(_N_STATES, -np.inf)
    pm[0] = 0.0
    choice = np.zeros((T, _N_STATES), np.int8)
    for t in range(T):
        branch = _OUT[:, :, 0] * soft[t, 0] + _OUT[:, :, 1] * soft[t, 1]  # (state, bit)
        cand = pm[_PRED_S] + branch[_PRED_S, _PRED_B]  # (next_state, 2)
        c = np.argmax(cand, axis=1)
        choice[t] = c
        pm = cand[np.arange(_N_STATES), c]
    bits, s = np.zeros(T, np.uint8), 0
    for t in range(T - 1, -1, -1):
        c = choice[t, s]
        bits[t] = _PRED_B[s, c]
        s = _PRED_S[s, c]
    return bits[: T - (_K - 1)]


def _n_coded(payload_bits: int) -> int:
    """Code symbols per frame: (payload + CRC + tail) bits at rate 1/2 (172 for 64 bits)."""
    return 2 * (payload_bits + _CRC_BITS + _K - 1)


class _Codec:
    """Payload bits <-> interleaved +-1 code symbols, for one payload size."""

    def __init__(self, payload_bits: int):
        self.payload_bits = payload_bits
        self.n_coded = _n_coded(payload_bits)
        self._perm = np.random.default_rng(0xBED).permutation(self.n_coded)

    def encode(self, payload_bits: np.ndarray) -> np.ndarray:
        coded = _conv_encode(np.concatenate([payload_bits, _crc16(payload_bits)]))
        return (1.0 - 2.0 * coded[self._perm]).astype(np.float32)

    def decode(self, soft: np.ndarray) -> np.ndarray | None:
        """Interleaved soft symbols -> payload bits, or None if the CRC fails."""
        deint = np.empty_like(soft)
        deint[self._perm] = soft
        bits = _viterbi(deint)
        n = self.payload_bits
        payload, crc = bits[:n], bits[n:n + _CRC_BITS]
        return payload if np.array_equal(_crc16(payload), crc) else None


@functools.lru_cache(maxsize=None)
def _codec(payload_bits: int) -> _Codec:
    return _Codec(payload_bits)


# --- bed shaping and frame layout ------------------------------------------------------------
def _shaping_filters(bed_path: Path | None, ntaps: int = 257) -> tuple[np.ndarray, np.ndarray]:
    """(tx shaping FIR h, rx equalizer FIR g), both linear phase, `ntaps` long."""
    n = 2 * (ntaps - 1)
    f = np.fft.rfftfreq(n, 1 / _FS)
    if bed_path and Path(bed_path).exists():
        import soundfile as sf

        bed, _ = sf.read(str(bed_path), dtype="float32", always_2d=True)
        bed = bed[:, 0]
        frames = bed[: len(bed) // n * n].reshape(-1, n) * np.hanning(n)
        H = np.sqrt((np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(0))
    else:  # generic room tone: falls ~3 dB/octave
        H = 1 / np.sqrt(np.maximum(f, 100) / 100)
    band = np.clip((f - 200) / 100, 0, 1) * np.clip((3600 - f) / 100, 0, 1)  # soft 300-3500 Hz band
    H = H / H[(f > 300) & (f < 3400)].max()
    G = band * H / (H**2 + 1e-3)  # regularized inverse, phone band only

    def fir(spec):
        x = np.fft.fftshift(np.fft.irfft(spec, n))
        c = len(x) // 2
        return (x[c - ntaps // 2: c + ntaps // 2 + 1] * np.hanning(ntaps)).astype(np.float32)

    return fir(H * band), fir(G)


class _StreamFIR:
    def __init__(self, taps: np.ndarray):
        self._taps, self._hist = taps, np.zeros(len(taps) - 1, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._hist, x.astype(np.float32)])
        self._hist = buf[len(buf) - (len(self._taps) - 1):]
        return np.convolve(buf, self._taps, "valid").astype(np.float32)


def _gap_chips(o: DsssOptions) -> int:
    return int(round(o.gap_s * _FS))


def _frame_chips(o: DsssOptions) -> int:
    return o.sync_chips + _n_coded(o.payload_bits) * o.chips_per_symbol


def _codes(o: DsssOptions) -> tuple[np.ndarray, np.ndarray]:
    """(sync PN, data PN) shared by embedder and detector."""
    rng = np.random.default_rng(0x5EED)
    pm = np.array([-1.0, 1.0], np.float32)
    return rng.choice(pm, o.sync_chips), rng.choice(pm, _n_coded(o.payload_bits) * o.chips_per_symbol)


# --- embedder ------------------------------------------------------------------------------
class _DsssEmbedder(Embedder):
    sample_rate = _FS

    def __init__(self, payload_bits: np.ndarray, o: DsssOptions):
        sync, data = _codes(o)
        codec = _codec(o.payload_bits)
        sym = codec.encode(payload_bits)
        frame = [sync, (data.reshape(codec.n_coded, -1) * sym[:, None]).ravel()]
        gap = _gap_chips(o)
        if gap:  # filler uses its own PN, uncorrelated with sync and data codes
            frame.append(np.random.default_rng(0x6A9).choice(np.array([-1.0, 1.0], np.float32), gap)
                         if o.gap_fill == "noise" else np.zeros(gap, np.float32))
        self._chips = np.concatenate(frame)
        h, _ = _shaping_filters(o.bed_path)
        self._fir = _StreamFIR(h)
        self._unit = 1.0 / float(np.sqrt(np.sum(h**2)))  # shaped chips -> unit RMS
        self._o = o
        self._env = 10 ** (o.level_dbfs / 20)  # current bed RMS
        self._pos = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        return audio + self._bed(len(audio), audio)

    def _bed(self, n: int, speech: np.ndarray) -> np.ndarray:
        """n samples of bed; with speech_margin_db set, its level follows the speech."""
        idx = (self._pos + np.arange(n)) % len(self._chips)
        self._pos = (self._pos + n) % len(self._chips)
        y = self._fir(self._chips[idx]) * self._unit
        base = 10 ** (self._o.level_dbfs / 20)
        if self._o.speech_margin_db is None:
            return y * base
        # 10 ms envelope: fast attack, ~150 ms release, never below the base bed level
        gains = np.empty(n, np.float32)
        k = 10 ** (-self._o.speech_margin_db / 20)
        for i in range(0, n, 80):
            target = max(base, k * float(np.sqrt(np.mean(speech[i:i + 80] ** 2))))
            self._env = target if target > self._env else max(target, self._env * 0.94)
            gains[i:i + 80] = self._env
        return y * gains


# --- detector ------------------------------------------------------------------------------
class _DsssDetector(Detector):
    sample_rate = _FS

    _SYNC_Z = 7.0  # normalized-correlation z threshold (noise max over a few seconds ~4.5)
    _RESYNC_Z = 4.0  # lower threshold when re-confirming sync near a predicted position
    _TRACK_EVERY = 8  # symbols per timing re-check
    _NORM_BLOCK = 80  # 10 ms loudness normalization blocks

    def __init__(self, o: DsssOptions):
        self._o = o
        self._codec = _codec(o.payload_bits)
        self._n = self._codec.n_coded
        self._sync, data = _codes(o)
        self._data = data.reshape(self._n, o.chips_per_symbol)
        _, g = _shaping_filters(o.bed_path)
        self._eq_fir = _StreamFIR(g)
        self._eq = np.zeros(0, np.float32)  # equalized + loudness-normalized per 10 ms block
        self._eq_tail = np.zeros(0, np.float32)
        self._search_from = 0
        self._pending_sync: tuple | None = None  # (pos, z) or ("predict", pos, z)
        self._acc = np.zeros(self._n, np.float32)
        self._n_acc = 0
        self._blind = 0  # consecutive frames decoded from a prediction without a visible sync
        self._result: Detection | None = None
        self._reported = False

    def push(self, audio: np.ndarray) -> Detection | None:
        if self._result is None:
            # normalize each 10 ms block to unit RMS: loud speech then weighs no more than
            # quiet bed, so correlations are driven by the bed-only stretches
            buf = np.concatenate([self._eq_tail, self._eq_fir(audio.astype(np.float32))])
            n = len(buf) // self._NORM_BLOCK * self._NORM_BLOCK
            blocks = buf[:n].reshape(-1, self._NORM_BLOCK)
            rms = np.sqrt(np.mean(blocks**2, axis=1, keepdims=True))
            self._eq = np.concatenate([self._eq, (blocks / np.maximum(rms, 1e-5)).ravel()])
            self._eq_tail = buf[n:]
            self._process()
        if self._result is None or self._reported:
            return None
        self._reported = True
        return self._result

    # -- sync -------------------------------------------------------------------------------
    def _find_sync(self) -> tuple[int, float] | None:
        Ls = self._o.sync_chips
        seg = self._eq[self._search_from:]
        if len(seg) < Ls + 2 * _PACKET:
            return None
        n = 1 << int(np.ceil(np.log2(len(seg) + Ls)))
        corr = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(self._sync, n)), n)[: len(seg) - Ls + 1]
        e = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
        # floor at ~-100 dB RMS so digital silence (pre-roll, PCMU zero) can't divide rounding noise by ~0
        energy = np.sqrt(np.maximum(e[Ls:] - e[:-Ls], Ls * 1e-10))
        zs = corr / energy  # = normalized correlation * sqrt(Ls), i.e. in noise sigmas (sync norm is sqrt(Ls))
        zs[np.abs(zs) > 1.01 * np.sqrt(Ls)] = 0.0  # Cauchy-Schwarz bound; anything above is numerical junk
        hits = np.nonzero(np.abs(zs) > self._SYNC_Z)[0]  # negative peak = inverted polarity on the path
        if len(hits) == 0:
            self._search_from += len(seg) - Ls - _PACKET  # keep a little overlap
            return None
        first = hits[0]
        if first + _PACKET >= len(zs):  # peak region may still be growing; wait for more audio
            return None
        p = first + int(np.argmax(np.abs(zs[first:first + _PACKET])))
        return self._search_from + p, float(zs[p])

    def _confirm_sync(self, pred: int, z_prev: float) -> tuple[tuple[int, float], bool]:
        """(Best sync within +-2 packets of the predicted position, True) or (the prediction, False)."""
        Ls, best = self._o.sync_chips, (pred, z_prev)
        best_abs = self._RESYNC_Z
        for o in range(-2 * _PACKET, 2 * _PACKET + 1):
            st = pred + o
            if st < 0 or st + Ls > len(self._eq):
                continue
            seg = self._eq[st:st + Ls]
            z = float(np.dot(seg, self._sync) / (np.linalg.norm(seg) + 1e-12))
            if abs(z) > best_abs and np.sign(z) == np.sign(z_prev):
                best, best_abs = (st, z), abs(z)
        return best, best_abs > self._RESYNC_Z

    # -- data -------------------------------------------------------------------------------
    def _despread(self, p: int) -> tuple[np.ndarray, list, int] | None:
        """Soft symbols of the frame at sync position p, or None if more audio is needed."""
        Ns, Ls = self._o.chips_per_symbol, self._o.sync_chips
        soft, off, shifts = np.zeros(self._n, np.float32), 0, []

        def dots(k0, k1, o):
            start = p + Ls + k0 * Ns + o
            if start < 0 or start + (k1 - k0) * Ns > len(self._eq):
                return None
            seg = self._eq[start:start + (k1 - k0) * Ns].reshape(k1 - k0, Ns)
            return np.einsum("ij,ij->i", seg, self._data[k0:k1])

        def llr(d, k0, k1, o):
            # weight each symbol by 1/local power: symbols under loud speech are unreliable,
            # so they must not dominate the Viterbi metric (soft decision ~ y / noise variance)
            start = p + Ls + k0 * Ns + o
            seg = self._eq[start:start + (k1 - k0) * Ns].reshape(k1 - k0, Ns)
            return d / (np.mean(seg**2, axis=1) + 1e-12)

        for k0 in range(0, self._n, self._TRACK_EVERY):
            k1 = min(self._n, k0 + self._TRACK_EVERY)
            cur = dots(k0, k1, off)
            if cur is None:
                return None
            best_e, best_o = float(np.sum(cur**2)), off
            # whole packets: jitter buffer inserts/drops; +-1 sample: clock drift between endpoints
            for o, margin in ((off - _PACKET, 1.5), (off + _PACKET, 1.5), (off - 1, 1.15), (off + 1, 1.15)):
                d = dots(k0, k1, o)
                if d is not None and float(np.sum(d**2)) > margin * best_e:
                    best_e, best_o = float(np.sum(d**2)), o
            if best_o != off:
                shifts.append((k0, best_o - off))
                off = best_o
                cur = dots(k0, k1, off)
            soft[k0:k1] = llr(cur, k0, k1, off)
        return soft, shifts, off

    def _process(self):
        frame = _frame_chips(self._o)
        period = frame + _gap_chips(self._o)  # sync-to-sync distance
        while self._result is None:
            if self._pending_sync is None:
                found = self._find_sync()
                if found is None:
                    return
                self._pending_sync = found
            if self._pending_sync[0] == "predict":
                _, pred, z_prev = self._pending_sync
                if len(self._eq) < pred + self._o.sync_chips + 3 * _PACKET:
                    return
                self._pending_sync, seen = self._confirm_sync(pred, z_prev)
                self._blind = 0 if seen else self._blind + 1
            p, z = self._pending_sync
            if len(self._eq) < p + frame + 3 * _PACKET:
                return  # wait for the whole frame (+ room for timing shifts)
            out = self._despread(p)
            if out is None:
                return
            soft, shifts, off = out
            soft = np.sign(z) * soft / (np.sqrt(np.mean(soft**2)) + 1e-12)
            self._acc += soft
            self._n_acc += 1
            payload, used = self._codec.decode(soft), 1
            if payload is None and self._n_acc > 1:
                payload, used = self._codec.decode(self._acc), self._n_acc
            if payload is not None:
                self._result = Detection(
                    payload=_to_bytes(payload),
                    confidence=1.0,  # CRC-16 verified
                    at_sample=len(self._eq),
                    verified=True,
                    info={"sync_z": round(z, 1), "frames": used, "timing_shifts": [s for _, s in shifts]},
                )
                return
            self._search_from = max(0, p + period + off - 3 * _PACKET)
            if self._n_acc >= 3 or self._blind >= 2:
                # the lock is probably wrong (false sync, or timing lost): drop it and search afresh
                self._acc[:], self._n_acc, self._blind, self._pending_sync = 0.0, 0, 0, None
                continue
            # flywheel: the next frame starts one frame later (plus tracked shifts); confirm
            # locally if the sync is visible, otherwise trust the prediction and keep combining
            self._pending_sync = ("predict", p + period + off, z)
