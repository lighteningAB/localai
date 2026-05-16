"""Phase 0 step (a): confirm a usable compile target for outfit-swap.

Preferred target: Snapdragon 8s Gen 4 (Nothing Phone 3 class).
Acceptable fallback: Snapdragon 8 Gen 3 (same Hexagon V79 family).

Run before any compile script. Exits non-zero if neither is in the catalog so
CI / a wrapping shell script can refuse to proceed.
"""

from __future__ import annotations

import sys

import qai_hub as hub

# Ordered preference list. We compile for whatever's available in the Hub
# catalog at run time. The 8s Gen 4 (Nothing Phone 3) uses Hexagon V79 — the
# 8 Elite QRD compiles to a binary that runs on the same V79 NPU. Forward-
# compat across Hexagon generations is real for QAIRT 2.x (the existing
# pipeline already runs V73-compiled xororz binaries on the V79 chip).
PREFERENCE = [
    # The Nothing Phone 3 in hand reports Hexagon arch=V73
    # (`socModel=91 arch=73`) — Snapdragon 8 Gen 2 family. The existing
    # xororz UNet/VAE binaries (V73-compiled) load and run on this chip.
    # QCS8550 is the chipset-level proxy for 8 Gen 2; the phone-family
    # targets (Samsung Galaxy S23) pin to specific QAIRT versions that
    # the device's runtime won't accept.
    "QCS8550 (Proxy)",              # 8 Gen 2 chipset proxy (Hexagon V73)
    "Samsung Galaxy S23 (Family)",  # phone-level fallback
    "Snapdragon 8s Gen 4",          # original plan target (not in catalog)
    "Snapdragon 8 Elite QRD",       # V79 (wrong arch for this device)
]


def pick_target() -> str:
    devices = hub.get_devices()
    names = {d.name for d in devices}
    print(f"Hub returned {len(names)} devices.")

    for preferred in PREFERENCE:
        matches = [d for d in devices if preferred.lower() == d.name.lower()]
        if matches:
            chosen = matches[0].name
            print(f"OK — using {chosen!r}")
            return chosen

    # Substring fallback in case Hub renames anything.
    for preferred in PREFERENCE:
        matches = [d for d in devices if preferred.lower() in d.name.lower()]
        if matches:
            chosen = matches[0].name
            print(f"OK — substring match {chosen!r} for preference {preferred!r}")
            return chosen

    print(
        f"ERROR — none of {PREFERENCE!r} found in qai-hub Device catalog. "
        f"Sample of available device names:",
        file=sys.stderr,
    )
    for name in sorted(names)[:30]:
        print(f"  - {name}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    target = pick_target()
    # Single line on stdout so a shell wrapper can capture it.
    print(f"COMPILE_TARGET={target}")
