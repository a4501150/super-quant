#!/usr/bin/env python3
"""Client-side expert-placement policy for the Infernix server.

The engine provides the mechanism (an exclusive vram/ram/ssd placement of the
routed experts, applied at runtime); this script decides the numbers:

  --status                            print the live per-tier counts and bytes
  --shares vram=60%,ram=25%,ssd=15%   apply the given placement (dry-run first)
  --reserve-vram-gb 32 --reserve-ram-gb 24
                                      the largest VRAM share that leaves 32 GB of
                                      VRAM free for other programs, then the largest
                                      RAM share that leaves 24 GB of host RAM, the
                                      rest on the SSD tier

Standard library only. The server's URL: --url or INFERNIX_URL or
http://127.0.0.1:${INFERNIX_PORT:-8000}.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GB = 1024 * 1024 * 1024


URL = None


def base_url():
    if URL:
        return URL.rstrip("/")
    if os.environ.get("INFERNIX_URL"):
        return os.environ["INFERNIX_URL"].rstrip("/")
    port = os.environ.get("INFERNIX_PORT", "8000")
    return f"http://127.0.0.1:{port}"


def call(method, path, body=None):
    request = urllib.request.Request(
        base_url() + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        sys.exit(f"{method} {path} failed: {error.code} {detail}")
    except urllib.error.URLError as error:
        sys.exit(f"cannot reach the server at {base_url()}: {error.reason}")


def get_status():
    report = call("GET", "/admin/expert-placement")
    if not report.get("supported"):
        sys.exit("this model runs without the expert cache")
    return report


def apply_spec(spec, note=""):
    preview = call("POST", "/admin/expert-placement?dry_run=1", parse_spec_body(spec))
    placement = preview.get("placement", {})
    predicted = preview.get("predicted", {})
    print(f"dry run: vram={placement.get('vram')} ram={placement.get('ram')} "
          f"ssd={placement.get('ssd')} -> vram {predicted.get('vram_expert_bytes', 0) / GB:.1f} GiB "
          f"(free after {predicted.get('vram_free_after', 0) / GB:.1f} GiB), "
          f"ram {predicted.get('ram_expert_bytes', 0) / GB:.1f} GiB"
          + (f" -- {note}" if note else ""))
    if not predicted.get("fits", True):
        sys.exit("the placement does not fit; nothing applied")
    result = call("POST", "/admin/expert-placement", parse_spec_body(spec))
    print(f"applied: {json.dumps(result)}")


def parse_spec_body(spec):
    """An --experts spec string -> the POST body's tier terms."""
    body = {}
    for term in spec.split(","):
        tier, _, value = term.partition("=")
        tier, value = tier.strip(), value.strip()
        if not value:
            raise SystemExit(f"bad spec term: {term}")
        body[tier] = value if value.endswith("%") or value == "all" else int(value)
    return body


def reserve_plan(report, reserve_vram_gb, reserve_ram_gb):
    """The largest placements that leave the reserves free."""
    mechanism = report.get("mechanism", {})
    total = report.get("total_experts") or mechanism.get("total_experts", 0)
    frame_bytes = mechanism.get("vram", {}).get("frame_bytes", 0)
    record_bytes = mechanism.get("ram", {}).get("record_bytes", 0)
    if not total or not frame_bytes:
        sys.exit("the server reports no expert sizes; is this the Flash-Next model?")
    # VRAM: the frames may grow into what the current pool holds plus what is free.
    budget = report.get("mechanism", {}).get("vram_free", 0) + mechanism.get("vram", {}).get("bytes", 0)
    budget -= reserve_vram_gb * GB
    vram = max(1, min(total, int(budget // frame_bytes)))
    # RAM: the host bytes the reserve leaves, capped by the experts not in VRAM.
    ram_budget = max(0, mechanism.get("host_available", 0) - reserve_ram_gb * GB)
    ram = max(0, min(total - vram, int(ram_budget // record_bytes))) if record_bytes else 0
    return vram, ram, total


def print_status(report):
    mechanism = report.get("mechanism", {})
    total = mechanism.get("total_experts", 0)
    vram = mechanism.get("vram", {})
    ram = mechanism.get("ram", {})
    ssd = mechanism.get("ssd", {})
    print(f"total experts:     {total}")
    print(f"vram: {vram.get('experts', 0):>6} experts  {vram.get('bytes', 0) / GB:7.2f} GiB "
          f"({vram.get('pinned', 0)} pinned, frame {vram.get('frame_bytes', 0)} B)")
    print(f"ram:  {ram.get('experts', 0):>6} experts  in {ram.get('slots', 0)} slots "
          f"(record {ram.get('record_bytes', 0)} B)")
    print(f"ssd:  {ssd.get('experts', 0):>6} experts")
    print(f"vram free/total:   {mechanism.get('vram_free', 0) / GB:.1f} / "
          f"{mechanism.get('vram_total', 0) / GB:.1f} GiB")
    print(f"host available:    {mechanism.get('host_available', 0) / GB:.1f} GiB")
    if report.get("switching"):
        print(f"switching to {report.get('target')} frames "
              f"(fill {report.get('fill_seeded', 0)}/{report.get('fill_total', 0)})")


def main():
    global URL
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="the server's base URL")
    parser.add_argument("--status", action="store_true", help="print the live placement")
    parser.add_argument("--shares", metavar="SPEC",
                        help="apply vram=..,ram=..[,ssd=..] (shares or counts)")
    parser.add_argument("--reserve-vram-gb", type=float, help="GB of VRAM to leave other programs")
    parser.add_argument("--reserve-ram-gb", type=float, help="GB of host RAM to leave free")
    args = parser.parse_args()
    URL = args.url

    if args.status:
        print_status(get_status())
        return
    if args.shares:
        apply_spec(args.shares)
        return
    if args.reserve_vram_gb is not None or args.reserve_ram_gb is not None:
        reserve_vram = args.reserve_vram_gb if args.reserve_vram_gb is not None else 0.0
        reserve_ram = args.reserve_ram_gb if args.reserve_ram_gb is not None else 0.0
        report = get_status()
        vram, ram, total = reserve_plan(report, reserve_vram, reserve_ram)
        spec = f"vram={vram},ram={ram}"
        apply_spec(spec, note=f"reserve {reserve_vram:g} GB VRAM, {reserve_ram:g} GB RAM")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
