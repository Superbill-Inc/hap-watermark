# Watermarking Demo

Pluggable audio watermarking for phone calls. There are two schemes behind one SPI: Meta's AudioSeal and a DSSS "bed". Testbeds run either one through a simulated channel or a real Telnyx PSTN call.

## Watermark SPI (`watermark/`)

| module | contents |
|---|---|
| `watermark/spi.py` | **Interfaces only.** `Scheme` declares `name`, `Options` (a dataclass), `payload_bits`, `sample_rate`, `frame_samples`, `embedder(payload)`, `detector()` and `describe()`. `Embedder.process(block) -> block` is the sender side; `Detector.push(block) -> Detection or None` is the receiver side. `Detection` is the `payload`, `confidence`, `at_sample`, `verified`, `info` record. |
| `watermark/factory/` | **All factory code.** `scheme_factory.py` has `SchemeFactory`: `register(name, Class or "module:Class")`, `names()`, `create(name, options)`. `options.py` has `parse_opts` and the conversion from `--opt` strings to the `Options` dataclass. `builtins.py` lists the built-in implementations. The package exports `default_factory`, which has them registered. |
| `watermark/base.py` | `BaseScheme`: an optional base for implementations. It stores the options, checks payload size, and implements `describe()`. |
| `watermark/adapters.py` | `at_rate(obj, rate, scheme)` runs a scheme at a different pipeline rate. It keeps every added delay a whole number of the scheme's frames. |
| `watermark/util.py` | Payload helpers: `payload_from_hex`, `payload_hex`. |
| `watermark/impls/` | **Implementations,** one module per scheme. `watermark/impls/<name>.py` defines `<Name>Scheme` (registered as `"<name>"`) and `<Name>Options`. The embedder and detector classes are private. |

Built-in schemes:
- **`dsss`** (`watermark/impls/dsss.py`): a 64-bit payload by default (`--opt payload_bits=128` for 128 bits; any multiple of 8 up to 256) in a spread-spectrum room-tone bed, with the modem included. It finds sync by itself and verifies with CRC-16 (`verified=True`). Runs at 8 kHz.
- **`audioseal`** (`watermark/impls/audioseal.py`): a 16-bit payload carried by the speech itself (base AudioSeal model, streamed block by block). The detector needs the 320-sample grid (`grid_offset`, 86 on the Telnyx route). There's no checksum, so it reports after `confirm` agreeing windows (`verified=False`). Runs at 16 kHz.

To add a scheme, put `MyScheme(BaseScheme)` in `watermark/impls/mine.py` and add `"mine": "watermark.impls.mine:MyScheme"` to `BUILTIN_IMPLS` in `watermark/factory/builtins.py`. Alternatively call `watermark.default_factory.register("mine", MyScheme)` at runtime. Both testbeds then accept `--scheme mine`. Schemes are imported only when used, so `dsss` never loads torch.

## Testbeds

**Local round trip** through a simulated phone channel. The channel adds PCMU, an unknown sub-packet offset, jitter-buffer packet jumps, AGC and noise, inverted polarity, bot speech, clock drift and silence suppression. The output also reports CPU cost as a real-time factor:

    uv run python testbed_local.py --scheme dsss --opt speech_margin_db=20
    uv run python testbed_local.py --scheme audioseal --trials 5
    uv run python testbed_local.py --list-schemes

**Real Telnyx call** (see "PSTN leg" below for the call setup). `--wm on|off|toggle` turns the watermark on or off, or switches it every `--toggle-s` seconds and prints each switch:

    uv run python testbed_pstn.py --scheme dsss --duration 45
    uv run python testbed_pstn.py --scheme dsss --mic --playback --input-device "<headset>" --output-device "<headset>" --opt speech_margin_db=20 --wm toggle --toggle-s 10
    uv run python testbed_pstn.py --scheme audioseal --mic --playback --opt grid_offset=86

**Offline call test (fake Telnyx):**

    uv run python loopback_test.py --testbed --scheme dsss
    uv run python loopback_test.py --testbed --scheme audioseal --speech say --speech-duty 1 --opt grid_offset=86

The scripts below (`main.py`, `stream.py`, `pstn.py`) are the original AudioSeal-only demos. They still work, and the testbeds above supersede them.

# AudioSeal-only demos

Embeds a 16-bit integer payload in audio with Meta's AudioSeal and extracts it again.
It also checks whether the watermark survives a telephony channel (8 kHz G.711 PCMU).

Setup: `uv sync` (Python 3.12; models download to `~/.cache/audioseal` on first run).

## Mic → PCMU → extract

    uv run python main.py mic --payload 42351 --seconds 5

Pipeline: the mic records at 16 kHz → watermark with the payload → downsample to 8 kHz → G.711 mu-law encode → decode → upsample to 16 kHz → detect and print the payload.

- `--file x.wav` uses a file instead of the mic
- `--save DIR` writes every stage (`1_input.wav`, `2_watermarked.wav`, `3_channel_8k.pcmu`, `4_received_16k.wav`)

On first use, macOS asks to give your terminal microphone access. If the input is silent, the app warns you.

## Live streaming: mic → watermark → PCMU → extract → playback

    uv run python stream.py --payload 42351

This processes 40 ms blocks in real time. The status line shows mic level, PCMU packet count, watermarking time per block, and the live detection with the extracted payload. Press Ctrl+C to get a summary. Use headphones; speakers feed back into the mic.

- `--list-devices`, `--input-device NAME`, `--output-device NAME` pick the audio devices (index or name substring)
- `--no-playback`, `--duration N`, `--save DIR`
- `--file x.wav` streams a file in place of the mic
- `--no-watermark` runs a control without the watermark (expect no detection)
- `--misalign N` cuts the detector window N samples off the frame grid (see below)

Mic-to-ear latency is about 280 ms: a 40 ms block, 40 ms lookahead, and a jitter buffer for soxr's bursty streaming resampler.

### Caveats found while building this
- **The "streaming" AudioSeal checkpoint doesn't survive 8 kHz.** Its watermark is destroyed by the 4 kHz band-limit. The demo runs the base 16-bit model block by block, with 500 ms of past context and 40 ms of lookahead.
- **Detection needs frame alignment.** The detector decodes reliably only when its window starts on the watermark's 320-sample grid. Off by a few samples, the payload comes out as garbage, often with a high detection score. This demo is sample-aligned end to end, so it cuts windows on the grid. A real receiver would need its own sync step. A brute-force search over offsets was slow (several seconds per search) and locked onto the right payload only 5 of 12 times.

## Other commands

    uv run python main.py embed in.wav out.wav --payload 1234
    uv run python main.py detect out.wav
    uv run python main.py bench [in.wav] --save

`bench` uses macOS `say` speech when no file is given. It reports the detection probability and the decoded payload for: an unwatermarked baseline, volume change, noise at 30/20/10 dB, an 8 kHz resample, G.711 at 8 kHz, and a 1 s crop.

## PSTN leg over a real Telnyx call

The call tests need:
- **Two Telnyx numbers on one Call Control application.** That application's webhook URL must be `PUBLIC_URL` + `WEBHOOK_PATH`.
- **A tunnel** (e.g. ngrok) from `PUBLIC_URL` to `localhost:8765` (`--port`).
- **These environment variables** (or the matching flags):

| variable | flag | meaning |
|---|---|---|
| `TELNYX_API_KEY` | – | Telnyx API key |
| `TELNYX_PUBLIC_KEY` | – | Telnyx webhook signing public key (Ed25519). Pass `--no-verify` to skip verification. |
| `TELNYX_FROM` | `--from` | calling number (TX leg), E.164 |
| `TELNYX_TO` | `--to` | answered number (RX leg), E.164 |
| `TELNYX_CONNECTION_ID` | `--connection-id` | Call Control application id |
| `PUBLIC_URL` | `--public-url` | public https URL tunneled to `--port` |
| `WEBHOOK_PATH` | `--webhook-path` | webhook path on that URL (default `/webhooks`) |
| `DSSS_BED_PATH` | `--opt bed_path=` | optional room-tone recording whose spectrum the DSSS bed imitates |

    uv run python pstn.py --payload 42351 --duration 60

The demo places a call from `TELNYX_FROM` to `TELNYX_TO`. The webhook and media streams are served on `localhost:8765`, exposed at `PUBLIC_URL`.
- TX leg: mic → watermark → 8 kHz PCMU → one 20 ms packet per tick into the call's bidirectional media stream.
- RX leg: the webhook answers the incoming call with a media stream → PCMU → 16 kHz → extract payload → local playback.

Telnyx webhooks are verified with `TELNYX_PUBLIC_KEY` (Ed25519). The call hangs up after `--duration` seconds, on Ctrl+C, or when either leg hangs up. Telnyx also enforces a `time_limit_secs` as a safety net.

The demo refuses to start if the port is already in use. Stop whatever else serves that webhook while you run it.

Useful flags: `--file x.wav` loops a file instead of the mic, plus `--no-playback`, `--save DIR`, and `--input-device`/`--output-device`.

After the call, the summary measures the real TX→RX channel delay by cross-correlation and reports its phase on the 160-sample (8 kHz) packet grid. The live detector assumes phase 0, because a 20 ms packet is exactly one watermark frame. If the call shifts the phase, the summary prints the `--rx-offset` that would realign it.

## Bed watermark: 64-bit payload with built-in sync (no per-provider offsets)

`watermark/impls/dsss.py` carries a 64-bit payload in a designed room-tone "bed": spread-spectrum noise shaped like room tone (a reference recording's spectrum via `DSSS_BED_PATH`, else a generic shape), at −47 dBFS. It doesn't depend on the bot speaking, and it needs no GPU.
- **Frame (repeats):** a 0.26 s sync sequence, then 172 coded symbols. The payload gets CRC-16 and a rate-½ K=7 convolutional code, plus an interleaver. A frame lasts 2.84 s.
- **Decoder:** inverse shaping filter, then 10 ms loudness normalization, then a sync search (exact sample position, either polarity). Symbols are despread while timing is tracked in ±1-packet steps (jitter buffer) and ±1-sample steps (clock drift). Then soft Viterbi and the CRC check. A "flywheel" combines repeated frames when one fails.
- **Optional:** `speech_margin_db` raises the bed under the bot's speech (speech masks it).
- **Gap between frames:** `gap_s` (default 0) adds a data-free pause after every frame. By default the pause is filled with the same shaped noise, so the bed sounds continuous; `gap_fill=silence` makes it silent. It lets a detector that finishes or restarts at the end of a frame still catch the next sync. With `gap_s=1`, back-to-back detections come every 3.8 s instead of every 5.7 s (64-bit).
- **Payload size:** `payload_bits` (default 64). A 128-bit frame has 300 coded symbols instead of 172 and lasts 4.76 s instead of 2.84 s. In the local test (10 trials per scenario, 10 s window, speech boost at 20 dB), 128 bits was as reliable as 64 bits, with a median first decode of about 5.0 s instead of 3.0 s. `chips_per_symbol=80` brings the frame down to 3.26 s (about 3.4 s to decode) but is less robust under bot speech.

**Local round trip (simulated channel):**

    uv run python testbed_local.py --scheme dsss [--trials 20] [--opt speech_margin_db=20] [--only speech]

Sample results (20 trials, 10 s window, the decoder uses ~3–10 ms of CPU per call):

| channel | decoded | median time |
|---|---|---|
| PCMU + unknown phase / + jitter jumps / + AGC + noise / inverted / 50 ppm drift | 20/20 each | 3.2 s |
| bot speech 30% / 60% / continuous (bed follows speech at −20 dB) | 18 / 17 / 18 of 20 | 3.2 s |
| silence suppression at −45 dBFS | 0/20 (expected: the bed is replaced by comfort noise) | – |


**Real Telnyx call** (same setup as `pstn.py`; the decoder re-arms after each detection):

    uv run python testbed_pstn.py --scheme dsss --duration 45
    uv run python testbed_pstn.py --scheme dsss --speech say --speech-duty 0.6 --opt speech_margin_db=20

Results on 2026-11-16: 7/7 correct with the bot silent and 7/7 with 60% bot speech, 0 wrong. The first detection came 3.0 s into the call, and the jitter-buffer jumps were followed automatically.

**Offline (fake Telnyx):** `uv run python loopback_test.py --testbed --scheme dsss`. The fake network adds a 43-sample phase shift and drops one packet.
