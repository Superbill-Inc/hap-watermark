"""Offline test of testbed_pstn.py / pstn.py: fakes Telnyx REST, webhooks (signed) and media
streams; no real call. The fake network adds 3 packets of delay, a 43-sample sub-packet phase
(as measured on Telnyx) and drops one packet.

    uv run python loopback_test.py --testbed [testbed_pstn args, e.g. --scheme dsss --duration 10]
    uv run python loopback_test.py [--duration 10]          # legacy AudioSeal demo (pstn.py)
"""
import asyncio, base64, json, os, sys, time
import aiohttp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
# fake call settings (nothing leaves this machine)
FAKE_CALL = ["--port", "8799", "--public-url", "http://127.0.0.1:8799", "--webhook-path", "/webhooks",
             "--from", "+15550100001", "--to", "+15550100002", "--connection-id", "fake-connection"]
TESTBED = "--testbed" in sys.argv
if TESTBED:
    sys.argv.remove("--testbed")
    sys.argv = ["testbed_pstn", "--duration", "14"] + FAKE_CALL + sys.argv[1:]
else:
    import tempfile
    import soundfile as sf
    import audio_io
    _speech = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(_speech, audio_io.tts_sample(16000), 16000)
    sys.argv = ["pstn", "--file", _speech, "--no-playback", "--duration", "14", "--payload", "4321",
                "--rx-offset", "86"] + FAKE_CALL + sys.argv[1:]
priv = Ed25519PrivateKey.generate()
os.environ["TELNYX_API_KEY"] = "fake"
os.environ["TELNYX_PUBLIC_KEY"] = base64.b64encode(priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
if not TESTBED:
    import pstn  # noqa: E402

async def post_webhook(event, payload, bad_sig=False):
    body = json.dumps({"data": {"event_type": event, "payload": payload}}).encode()
    ts = str(int(time.time()))
    sig = base64.b64encode(priv.sign(f"{ts}|".encode() + body)).decode()
    if bad_sig: sig = base64.b64encode(b"x" * 64).decode()
    async with aiohttp.ClientSession() as s:
        async with s.post("http://127.0.0.1:8799/webhooks", data=body,
                          headers={"telnyx-signature-ed25519": sig, "telnyx-timestamp": ts}) as r:
            return r.status

async def relay(demo):
    await asyncio.sleep(0.5)
    print("bad-signature webhook ->", await post_webhook("call.initiated", {}, bad_sig=True))
    await post_webhook("call.initiated", {"call_control_id": "rx-1", "direction": "incoming",
                                          "from": demo.args.from_number, "to": demo.args.to_number})
    async with aiohttp.ClientSession() as s:
        tx = await s.ws_connect("ws://127.0.0.1:8799/ws/tx"); rx = await s.ws_connect("ws://127.0.0.1:8799/ws/rx")
        fmt = {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}
        await tx.send_str(json.dumps({"event": "start", "start": {"media_format": fmt}}))
        await rx.send_str(json.dumps({"event": "start", "start": {"media_format": fmt}}))
        # network: 3 leading silent packets, then forward everything; drop packet #200 to test gap fill
        chunk = 0
        for _ in range(3):
            chunk += 1
            await rx.send_str(json.dumps({"event": "media", "media": {"track": "inbound", "chunk": str(chunk), "payload": base64.b64encode(b"\xff" * 160).decode()}}))
        # re-chunk 43 samples off the sender's packet grid (the sub-packet phase measured on Telnyx)
        buf = b"\xff" * 43
        async for m in tx:
            buf += base64.b64decode(json.loads(m.data)["media"]["payload"])
            while len(buf) >= 160:
                pkt, buf = buf[:160], buf[160:]
                chunk += 1
                if chunk == 200: continue
                await rx.send_str(json.dumps({"event": "media", "media": {"track": "inbound", "chunk": str(chunk), "payload": base64.b64encode(pkt).decode()}}))

async def main():
    if TESTBED:
        import testbed_pstn
        demo = testbed_pstn.WatermarkCall(pstn_args)
    else:
        demo = pstn.Demo(pstn_args)
    calls = []
    async def fake_telnyx(path, body):
        calls.append((path, {k: v for k, v in body.items() if k in ("from", "to", "stream_url", "stream_track", "stream_bidirectional_codec")}))
        if path == "/calls":
            asyncio.get_running_loop().create_task(relay(demo))
            return {"data": {"call_control_id": "tx-1"}}
        if path.endswith("/hangup"): demo.done.set()
        return {"data": {}}
    demo.telnyx = fake_telnyx
    await demo.run()
    print("REST calls:"); [print(" ", c) for c in calls]

if TESTBED:
    import testbed_pstn
    pstn_args = testbed_pstn.parse_args(sys.argv[1:])
else:
    pstn_args = pstn.parse_args(sys.argv[1:])
asyncio.run(main())
