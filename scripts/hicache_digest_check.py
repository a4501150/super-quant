#!/usr/bin/env python3
"""HiCache benchmark digest parsing and probe variance classification.

Parses the digest lines that the locally patched SGLang HiCache path emits
under SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS:
  * "HiCache page digest ..." / "HiCache storage tensor digest ..." (v1)
    record storage writes and reads per pool component;
  * "HiCache KV transfer digest ..." (MHA pool, device_index=...) and
    "HiCache transfer digest ..." (mamba pool, device_slot=...) record every
    host<->device copy per pool component (kv_k, kv_v, kv_scale_k,
    kv_scale_v, recurrent state). exact=False is always a hard failure.

Also classifies a first L3 restore marker omission as generation variance
only when storage plus transfer digests are exact and the immediate L1
replay reproduces the complete expected marker sequence.
"""

import json
import re
import sys

PAGE_RE = re.compile(
    r"HiCache page digest direction=(\w+) pool=(\S+) component=(\S+) key=(\S+) "
    r"host_page=(\d+) bytes=(\d+) sha256=([0-9a-f]{64})"
)
STORAGE_V1_RE = re.compile(
    r"HiCache storage tensor digest direction=(\w+) key=(\S+) bytes=(\d+) "
    r"sha256=([0-9a-f]{64})"
)
TRANSFER_RE = re.compile(
    r"HiCache (?:KV )?transfer digest direction=(\w+) component=(\S+) "
    r"host_page=(\d+) device_\w+=\d+ bytes=(\d+) host_sha256=([0-9a-f]{64}) "
    r"device_sha256=([0-9a-f]{64}) exact=(True|False)"
)

MARKER_ERROR_PREFIXES = (
    "markers missing from response",
    "response marker sequence mismatch",
)
REQUIRED_KV_TRANSFER_COMPONENTS = {"kv_k", "kv_v", "kv_scale_k", "kv_scale_v"}


def load_storage_digests(path):
    """Return {direction: {identity: set(digests)}} plus a matched-line count."""
    records = {}
    lines = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "HiCache" not in line or "digest" not in line:
                    continue
                m = PAGE_RE.search(line)
                if m:
                    direction = m.group(1)
                    identity = (m.group(2), m.group(3), m.group(4))
                    digest = m.group(7)
                else:
                    m = STORAGE_V1_RE.search(line)
                    if not m:
                        continue
                    direction = m.group(1)
                    identity = ("v1", "", m.group(2))
                    digest = m.group(4)
                lines += 1
                records.setdefault(direction, {}).setdefault(identity, set()).add(digest)
    except FileNotFoundError:
        pass
    return records, lines


def load_transfer_digests(path):
    """Parse KV/mamba host<->device transfer digest records from a server log."""
    records = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "transfer digest" not in line:
                    continue
                m = TRANSFER_RE.search(line)
                if not m:
                    continue
                records.append(
                    {
                        "direction": m.group(1),
                        "component": m.group(2),
                        "host_page": int(m.group(3)),
                        "bytes": int(m.group(4)),
                        "host_sha256": m.group(5),
                        "device_sha256": m.group(6),
                        "exact": m.group(7) == "True",
                    }
                )
    except FileNotFoundError:
        pass
    return records


def _rows(items):
    return [
        {
            "pool": k[0],
            "component": k[1],
            "key": k[2],
            **{name: sorted(values) for name, values in detail.items()},
        }
        for k, detail in sorted(items.items(), key=lambda item: str(item[0]))[:50]
    ]


def _conflict_rows(items, direction):
    return [
        {"pool": k[0], "component": k[1], "key": k[2], direction: sorted(values)}
        for k, values in sorted(items.items(), key=lambda item: str(item[0]))[:50]
    ]


def digest_check(cold_log, restore_log):
    """Compare cold write hashes against restore read hashes and check transfers.

    exact=False on any KV/mamba transfer digest is a hard failure. The dynamic
    K, V, K-scale, and V-scale components must all be present. Inconsistent
    writes or reads, mismatched pairs, and contradictory exact flags also fail.
    """
    cold, cold_lines = load_storage_digests(cold_log)
    restore, restore_lines = load_storage_digests(restore_log)
    writes = cold.get("write", {})
    reads = restore.get("read", {})
    paired = sorted(set(writes) & set(reads), key=str)
    write_conflicts = {k: v for k, v in writes.items() if len(v) != 1}
    read_conflicts = {k: v for k, v in reads.items() if len(v) != 1}
    mismatched = {k: {"write": writes[k], "read": reads[k]}
                  for k in paired if writes[k] != reads[k]}

    transfers = load_transfer_digests(cold_log) + load_transfer_digests(restore_log)
    exact_false = [r for r in transfers if not r["exact"]]
    flag_conflicts = [r for r in transfers
                      if (r["host_sha256"] == r["device_sha256"]) != r["exact"]]
    components = {}
    for record in transfers:
        components[record["component"]] = components.get(record["component"], 0) + 1

    missing_components = sorted(REQUIRED_KV_TRANSFER_COMPONENTS - components.keys())
    ok = (
        bool(paired)
        and not write_conflicts
        and not read_conflicts
        and not mismatched
        and not exact_false
        and not flag_conflicts
        and not missing_components
    )
    out = {
        "ok": ok,
        "cold_digest_lines": cold_lines,
        "restore_digest_lines": restore_lines,
        "write_keys": len(writes),
        "read_keys": len(reads),
        "paired_keys": len(paired),
        "unpaired_read_keys": len(set(reads) - set(writes)),
        "write_conflict_count": len(write_conflicts),
        "read_conflict_count": len(read_conflicts),
        "mismatch_count": len(mismatched),
        "transfer_records": len(transfers),
        "transfer_exact_false": len(exact_false),
        "transfer_flag_conflicts": len(flag_conflicts),
        "transfer_components": components,
        "missing_transfer_components": missing_components,
        "write_conflicts": _conflict_rows(write_conflicts, "writes"),
        "read_conflicts": _conflict_rows(read_conflicts, "reads"),
        "mismatched_keys": _rows(mismatched),
        "transfer_failures": [
            dict(r) for r in (exact_false + flag_conflicts)[:50]
        ],
    }
    if not ok:
        out["error"] = (
            f"digest check failed: paired={len(paired)}, "
            f"write_conflicts={len(write_conflicts)}, "
            f"read_conflicts={len(read_conflicts)}, mismatched={len(mismatched)}, "
            f"transfer_exact_false={len(exact_false)}, "
            f"transfer_flag_conflicts={len(flag_conflicts)}, "
            f"missing_transfer_components={','.join(missing_components) or 'none'}"
        )
    return out


def omission_only(metrics):
    """True when a failed probe's only marker defect is omission.

    The reported markers must form a strict in-order subsequence of the
    expected sequence: any foreign marker or reordering keeps the probe a
    hard failure.
    """
    expected = metrics.get("expected_markers") or []
    reported = metrics.get("response_markers") or []
    if not expected or len(reported) >= len(expected):
        return False
    pos = -1
    for marker in reported:
        rest = expected[pos + 1:]
        if marker not in rest:
            return False
        pos = pos + 1 + rest.index(marker)
    return True


def replay_is_complete(replay):
    """True when the immediate L1 replay reproduced every expected marker."""
    return bool(replay) and bool(replay.get("ok")) and replay.get("matched") is True


def _is_marker_error(metrics):
    return metrics.get("error", "").startswith(MARKER_ERROR_PREFIXES)


def classify_generation_variance(metrics, replay, digests):
    """Classify a failed L3 restore as generation variance, or None.

    Variance requires: omission-only marker defect, exact storage plus KV
    transfer digests (digests.ok), and an immediate L1 replay with the
    complete expected marker sequence. Repeated omission, foreign markers,
    digest conflicts/mismatches, or exact=False remain failures.
    """
    if metrics is None or metrics.get("ok"):
        return None
    if not _is_marker_error(metrics):
        return None
    if not omission_only(metrics):
        return None
    if not digests or not digests.get("ok"):
        return None
    if not replay_is_complete(replay):
        return None
    reported = set(metrics.get("response_markers") or [])
    return {
        "classification": "generation_variance",
        "reason": (
            "first L3 restore omitted markers while storage and KV transfer "
            "digests were exact, and the immediate L1 replay reproduced every "
            "expected marker"
        ),
        "missing_markers": [
            m for m in metrics.get("expected_markers") or [] if m not in reported
        ],
    }


def needs_replay(metrics):
    """True when a failed L3 restore should get an immediate L1 replay."""
    return (
        not metrics.get("ok")
        and _is_marker_error(metrics)
        and omission_only(metrics)
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: hicache_digest_check.py check <cold> <restore> <out-json> | needs-replay <metrics-json>")
    command = argv[0]
    if command == "check":
        cold_log, restore_log, out_path = argv[1:4]
        out = digest_check(cold_log, restore_log)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}),
              file=sys.stderr)
        return 0 if out["ok"] else 1
    if command == "needs-replay":
        try:
            with open(argv[1]) as f:
                metrics = json.load(f)
        except Exception:  # noqa: BLE001
            return 1
        return 0 if needs_replay(metrics) else 1
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
