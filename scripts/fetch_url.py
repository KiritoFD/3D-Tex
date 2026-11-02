#!/usr/bin/env python3
"""Multi-connection HTTP Range downloader with resume, for slow single-stream hosts.

Several dataset hosts (and any CDN that rate-limits per connection) deliver ~40 MB/min on one
socket but ~170 MB/min on six. This does the split/merge itself so it works for arbitrary
archives, not just the ones a specific dataset script knows about.

    python3 scripts/fetch_url.py --url https://host/file.zip --out data/.cache/file.zip -n 8
    python3 scripts/fetch_url.py --url ... --insecure      # self-signed / broken chain hosts
"""
from __future__ import annotations

import argparse
import json
import ssl
import threading
import time
import urllib.request
from pathlib import Path


def context_for(insecure: bool) -> ssl.SSLContext | None:
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def probe(url: str, timeout: int, ctx) -> tuple[int | None, bool]:
    """Return (total_bytes, honours_range)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "3D-Tex/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        ranged = response.status == 206
        total = None
        cr = response.headers.get("Content-Range", "")
        if "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
        elif response.headers.get("Content-Length"):
            total = int(response.headers["Content-Length"])
        response.read(1)
        return total, ranged


def plan(total: int, connections: int) -> list[tuple[int, int]]:
    size = total // connections + 1
    spans = []
    for index in range(connections):
        start = index * size
        if start >= total:
            break
        spans.append((start, min(total, start + size)))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel resumable HTTP download")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", "--connections", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ctx = context_for(args.insecure)
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    state_path = dest.with_name(dest.name + ".part.json")

    if dest.is_file() and dest.stat().st_size:
        print(f"[cache] {dest.name}: {dest.stat().st_size / 2**20:.1f} MB")
        return

    total, ranged = probe(args.url, args.timeout, ctx)
    if total is None:
        raise SystemExit("server did not report a size")
    spans = plan(total, args.connections if ranged else 1)
    print(f"[plan] {total / 2**20:.1f} MB over {len(spans)} connection(s)"
          f"{'' if ranged else ' (server ignores Range)'}")

    if part.is_file() and part.stat().st_size != total:
        part.unlink()
    if not part.is_file():
        with part.open("wb") as handle:
            handle.truncate(total)

    done = [0] * len(spans)
    if state_path.is_file():
        try:
            saved = json.loads(state_path.read_text())
            if saved.get("total") == total and len(saved.get("done", [])) == len(spans):
                done = list(saved["done"])
        except (ValueError, OSError):
            pass
    lock = threading.Lock()
    started = time.time()

    def persist() -> None:
        state_path.write_text(json.dumps({"url": args.url, "total": total, "done": list(done)}))

    def worker(index: int, start: int, end: int) -> None:
        pos = start + done[index]
        if pos >= end:
            return
        while pos < end:
            req = urllib.request.Request(args.url, headers={
                "Range": f"bytes={pos}-{end - 1}", "User-Agent": "3D-Tex/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=args.timeout, context=ctx) as response:
                    if response.status != 206:
                        raise RuntimeError("server stopped honouring Range")
                    with part.open("r+b") as handle:
                        handle.seek(pos)
                        while pos < end:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            pos += len(chunk)
                            with lock:
                                done[index] = pos - start
            except Exception as exc:            # retry the remainder in this segment
                print(f"  [retry seg{index}] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                time.sleep(2)

    for round_index in range(args.rounds):
        threads = [threading.Thread(target=worker, args=(i, a, b), daemon=True)
                   for i, (a, b) in enumerate(spans)]
        for thread in threads:
            thread.start()
        while any(t.is_alive() for t in threads):
            time.sleep(3)
            persist()
            got = sum(done)
            rate = got / 2**20 / max(time.time() - started, 1e-6) * 60
            print(f"  {got / 2**20:6.1f}/{total / 2**20:.1f} MB ({rate:.0f} MB/min)", flush=True)
        if sum(done) >= total:
            break

    if sum(done) < total:
        raise SystemExit(f"incomplete: {sum(done)}/{total}; rerun to resume")
    part.replace(dest)
    state_path.unlink(missing_ok=True)
    seconds = time.time() - started
    print(f"[ok] {dest} ({dest.stat().st_size / 2**20:.1f} MB in {seconds:.0f}s)")
    if args.json:
        print(json.dumps({"path": str(dest), "bytes": dest.stat().st_size,
                          "seconds": round(seconds, 1)}))


if __name__ == "__main__":
    main()
