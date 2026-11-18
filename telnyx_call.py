"""Telnyx call plumbing for the watermark testbeds (no ML dependencies).

TelnyxCall places a call FROM -> TO on a Call Control app whose webhook is served here
(exposed via ngrok as --public-url), answers the incoming leg itself, and wires both legs'
bidirectional media streams to this process:
  - TX leg: subclasses fill self.tx_q with 160-byte PCMU packets; one is sent every 20 ms
  - RX leg: received packets (gaps refilled with silence) go to on_rx_audio()
Webhooks are verified with TELNYX_PUBLIC_KEY (Ed25519).
"""

import argparse
import asyncio
import base64
import json
import os
import queue
import socket
import sys
import threading
import time

import aiohttp
from aiohttp import web

PCMU_SR = 8000
PACKET = 160  # 20 ms of 8 kHz PCMU == one 320-sample watermark frame at 16 kHz
SILENCE = b"\xff" * PACKET
TELNYX_API = "https://api.telnyx.com/v2"
SIG_TOLERANCE_S = 300


def log(msg: str) -> None:
    sys.stdout.write("\r" + " " * 120 + "\r" + msg + "\n")
    sys.stdout.flush()


class WebhookVerifier:
    """Ed25519 check of Telnyx webhooks: signature over f"{timestamp}|{raw_body}"."""

    def __init__(self, public_key_b64: str | None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        self.key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64)) if public_key_b64 else None

    def valid(self, body: bytes, signature: str | None, timestamp: str | None) -> bool:
        if self.key is None:
            return True
        try:
            if abs(time.time() - int(timestamp)) > SIG_TOLERANCE_S:
                return False
            self.key.verify(base64.b64decode(signature), f"{timestamp}|".encode() + body)
            return True
        except Exception:
            return False


class TelnyxCall:
    """Call plumbing shared by the demos: webhook (signed), dial/answer with media streams,
    paced 20 ms PCMU TX packets from self.tx_q, RX packets with gap fill -> on_rx_audio()."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = os.environ["TELNYX_API_KEY"]
        self.verifier = WebhookVerifier(None if args.no_verify else os.environ.get("TELNYX_PUBLIC_KEY"))
        self.tx_q: queue.Queue = queue.Queue()  # 160-byte PCMU packets to send
        self.tx_active = threading.Event()
        self.last_chunk: int | None = None
        self.tx_call_id: str | None = None
        self.rx_call_id: str | None = None
        self.done = asyncio.Event()
        self.sockets: set[web.WebSocketResponse] = set()
        self.stop = threading.Event()
        self.sent_log: list[bytes] = []
        self.rx_log: list[bytes] = []
        self.status = "dialing..."
        self.stats = dict(tx_pkts=0, tx_fill=0, tx_drop=0, rx_pkts=0, rx_gap=0)

    # --- Telnyx REST -----------------------------------------------------------------
    async def telnyx(self, path: str, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with self.http.post(f"{TELNYX_API}{path}", json=body, headers=headers) as r:
            data = await r.json(content_type=None)
            if r.status >= 300:
                raise RuntimeError(f"Telnyx {path} -> {r.status}: {data}")
            return data

    def ws_url(self, path: str) -> str:
        return self.args.public_url.replace("https://", "wss://").rstrip("/") + path

    def stream_params(self, path: str) -> dict:
        return {
            "stream_url": self.ws_url(path),
            "stream_track": "inbound_track",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
        }

    async def dial(self):
        body = {
            "connection_id": self.args.connection_id,
            "from": self.args.from_number,
            "to": self.args.to_number,
            "client_state": base64.b64encode(b"watermark-demo-tx").decode(),
            "time_limit_secs": int(self.args.duration) + 30,  # safety net if we crash
            **self.stream_params("/ws/tx"),
        }
        data = await self.telnyx("/calls", body)
        self.tx_call_id = data["data"]["call_control_id"]
        log(f"[call] dialing {self.args.from_number} -> {self.args.to_number}")

    async def hangup(self):
        for cid in (self.tx_call_id, self.rx_call_id):
            if cid:
                try:
                    await self.telnyx(f"/calls/{cid}/actions/hangup", {})
                except Exception as exc:  # already gone
                    if "90018" not in str(exc) and "already ended" not in str(exc).lower():
                        log(f"[call] hangup {cid[:12]}...: {exc}")

    # --- webhook ---------------------------------------------------------------------
    async def webhook(self, request: web.Request) -> web.Response:
        body = await request.read()
        if not self.verifier.valid(body, request.headers.get("telnyx-signature-ed25519"),
                                   request.headers.get("telnyx-timestamp")):
            log("[webhook] rejected: bad signature")
            return web.Response(status=401)
        data = json.loads(body).get("data", {})
        event, p = data.get("event_type"), data.get("payload", {})
        cid = p.get("call_control_id")

        if event == "call.initiated" and p.get("direction") == "incoming":
            if p.get("from") == self.args.from_number and p.get("to") == self.args.to_number and not self.rx_call_id:
                self.rx_call_id = cid
                log("[webhook] incoming demo call -> answering with media stream")
                asyncio.create_task(self.answer(cid))
            else:
                log(f"[webhook] ignoring unrelated incoming call event ({event})")
        elif cid in (self.tx_call_id, self.rx_call_id):
            leg = "TX" if cid == self.tx_call_id else "RX"
            if event in ("call.answered", "streaming.started", "streaming.stopped", "streaming.failed"):
                log(f"[webhook] {leg} {event}")
            elif event == "call.hangup":
                log(f"[webhook] {leg} call.hangup ({p.get('hangup_cause')}, {p.get('hangup_source')})")
                self.done.set()
        return web.Response(text="ok")

    async def answer(self, cid: str):
        try:
            await self.telnyx(f"/calls/{cid}/actions/answer", {
                **self.stream_params("/ws/rx"),
                "client_state": base64.b64encode(b"watermark-demo-rx").decode(),
            })
        except Exception as exc:
            log(f"[call] answer failed: {exc}")
            self.done.set()

    # --- TX media stream (we send watermarked audio into the call) ----------------------
    async def ws_tx(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.add(ws)
        sender = None
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            if m.get("event") == "start":
                log(f"[tx] stream started, format {m.get('start', {}).get('media_format')}")
                self.tx_active.set()
                sender = asyncio.create_task(self.tx_sender(ws))
            elif m.get("event") == "stop":
                break
        self.tx_active.clear()
        if sender:
            sender.cancel()
        return ws

    async def tx_sender(self, ws: web.WebSocketResponse):
        """Send one 20 ms packet per tick. Fill underruns and trim overruns in whole packets,
        which keeps the 8 kHz stream on the 160-sample (= 320 @ 16 kHz) watermark grid."""
        while self.tx_q.qsize() < self.args.tx_prebuffer:
            await asyncio.sleep(0.005)
        t0, n = time.monotonic(), 0
        while not ws.closed and not self.stop.is_set():
            while self.tx_q.qsize() > 3 * self.args.tx_prebuffer:  # bound latency
                self.tx_q.get_nowait()
                self.stats["tx_drop"] += 1
            try:
                pkt = self.tx_q.get_nowait()
            except queue.Empty:
                pkt = SILENCE
                self.stats["tx_fill"] += 1
            await ws.send_str(json.dumps({"event": "media", "media": {"payload": base64.b64encode(pkt).decode()}}))
            self.sent_log.append(pkt)
            self.stats["tx_pkts"] += 1
            n += 1
            await asyncio.sleep(max(0.0, t0 + n * 0.02 - time.monotonic()))

    # --- RX media stream (audio arriving at the answered number) ---------------------------
    async def ws_rx(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.add(ws)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            ev = m.get("event")
            if ev == "start":
                log(f"[rx] stream started, format {m.get('start', {}).get('media_format')}")
            elif ev == "media":
                media = m.get("media", {})
                if media.get("track", "inbound") != "inbound":
                    continue
                self.rx_packet(base64.b64decode(media["payload"]), media.get("chunk"))
            elif ev == "stop":
                break
        return ws

    def rx_packet(self, data: bytes, chunk):
        # refill lost packets with silence so the grid stays aligned (packets are 20 ms each)
        if chunk is not None and len(data) == PACKET:
            chunk = int(chunk)
            if self.last_chunk is not None and chunk > self.last_chunk + 1:
                gap = chunk - self.last_chunk - 1
                self.stats["rx_gap"] += gap
                data = SILENCE * gap + data
            self.last_chunk = chunk
        self.rx_log.append(data)
        self.stats["rx_pkts"] += 1
        self.on_rx_audio(data)

    def on_rx_audio(self, data: bytes):
        """Received PCMU (whole 20 ms packets, gaps refilled). Override."""

    async def serve_call(self, on_ready=None, on_teardown=None):
        """Start the webhook/media server, dial, wait for hangup / duration / Ctrl+C, clean up."""
        a = self.args
        app = web.Application()
        app.router.add_post(a.webhook_path, self.webhook)
        app.router.add_get("/ws/tx", self.ws_tx)
        app.router.add_get("/ws/rx", self.ws_rx)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=2)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", a.port).start()
        print(f"Webhook {a.public_url}{a.webhook_path} -> localhost:{a.port}"
              f" (signature check {'on' if self.verifier.key else 'OFF'})")
        if on_ready:
            on_ready()
        self.http = aiohttp.ClientSession()
        try:
            await self.dial()
            try:
                await asyncio.wait_for(self.done.wait(), timeout=a.duration)
            except asyncio.TimeoutError:
                log(f"[call] {a.duration:.0f}s limit reached")
        except asyncio.CancelledError:
            pass
        finally:
            self.stop.set()  # stops the TX sender and worker threads
            await self.hangup()
            await self.http.close()
            for ws in list(self.sockets):
                await ws.close()
            if on_teardown:
                on_teardown()
            await runner.cleanup()


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def add_call_args(p: argparse.ArgumentParser) -> None:
    """Telnyx call options shared by pstn.py and testbed_pstn.py (defaults come from the environment)."""
    env = os.environ.get
    p.add_argument("--from", dest="from_number", default=env("TELNYX_FROM"),
                   help="calling number, E.164 (TX leg) [env TELNYX_FROM]")
    p.add_argument("--to", dest="to_number", default=env("TELNYX_TO"),
                   help="answered number, E.164 (RX leg) [env TELNYX_TO]")
    p.add_argument("--connection-id", default=env("TELNYX_CONNECTION_ID"),
                   help="Telnyx Call Control application id; both numbers must be on it [env TELNYX_CONNECTION_ID]")
    p.add_argument("--public-url", default=env("PUBLIC_URL"),
                   help="public https URL tunneled to --port, e.g. an ngrok domain [env PUBLIC_URL]")
    p.add_argument("--webhook-path", default=env("WEBHOOK_PATH", "/webhooks"),
                   help="path of the Call Control app's webhook URL [env WEBHOOK_PATH, default /webhooks]")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--duration", type=float, default=60, help="hang up after N seconds")
    p.add_argument("--tx-prebuffer", type=int, default=5, help="packets buffered before sending starts")
    p.add_argument("--no-verify", action="store_true", help="skip Telnyx webhook signature verification")
    p.add_argument("--save", metavar="DIR", help="save sent/received audio after the call")


def check_call_env(args: argparse.Namespace) -> None:
    missing = [f"{flag} / {var}" for flag, var, value in (
        ("--from", "TELNYX_FROM", args.from_number), ("--to", "TELNYX_TO", args.to_number),
        ("--connection-id", "TELNYX_CONNECTION_ID", args.connection_id), ("--public-url", "PUBLIC_URL", args.public_url),
    ) if not value]
    if missing:
        sys.exit("error: call settings missing; pass the flag or set the env var: " + ", ".join(missing))
    if "TELNYX_API_KEY" not in os.environ:
        sys.exit("error: TELNYX_API_KEY not set")
    if not args.no_verify and "TELNYX_PUBLIC_KEY" not in os.environ:
        sys.exit("error: TELNYX_PUBLIC_KEY not set (needed to verify webhooks; or pass --no-verify)")
    if not port_free(args.port):
        sys.exit(f"error: port {args.port} is in use (another process is serving it; stop it or pick --port)")
