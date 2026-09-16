#!/usr/bin/env python3
"""Keep every visible GPU above the Hendrix idle-downscale threshold.

Hendrix reclaims GPU worker groups whose AVERAGE utilization stays below
2.5% for an hour (event reason ``GPUIdleDownscaling`` — it killed the FSDP2
preflight mid-calibration, because dispatch-bound passes keep SMs mostly
idle). This guard launches a tiny kernel on each device with a high duty
cycle: utilization is a any-kernel-active sample, so a launch every
``--period-ms`` reads as steady activity while costing negligible SM time.

Run alongside calibration; it is memory-frugal (one 256x256 bf16 matrix per
device) and exits with the process group (start it with the launcher, kill
it when the run ends).
"""

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-ms", type=float, default=2.0)
    args = ap.parse_args()

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    vals = {
        d: torch.randn(256, 256, device=d, dtype=torch.bfloat16) for d in devices
    }
    while True:
        for d, v in vals.items():
            v @= v.to(v.dtype) * 1e-9  # tiny kernel; overflow-safe rescale
        for d in devices:
            torch.cuda.synchronize(d)
        time.sleep(args.period_ms / 1000.0)


if __name__ == "__main__":
    main()
