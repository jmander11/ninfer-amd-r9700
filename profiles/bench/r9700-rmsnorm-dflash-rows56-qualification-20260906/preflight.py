#!/usr/bin/env python3
"""CPU-only immutable-input validation for the rows5/6 screen."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-rmsnorm-dflash-rows56-qualification-20260906"
EXPECTED = {
    "tools/r9700/build/rmsnorm_dflash_rows56_qual": "c898d101e970a4a8898663743e84c338b9758180b401996dd04a1d4ebbef8315",
    "tools/r9700/build/eager_ops.s": "2ebc597ecc05d0daa103d293a31f6e66d6edec4cc68ed4aa6fc18fa544049f92",
    "tools/r9700/rmsnorm_dflash_rows56_qual.hip": "6a1d4426cb23429f2f3106a28f65c225a56cc93fb2bcdadc5c564ccf4f635729",
    "src/ops/r9700/eager/eager_ops.h": "bf0694de7112c09d97aa49602e80116a6460e008b55f5e73d6f03603429a479e",
    "src/ops/r9700/eager/eager_ops.hip": "5fdb1cbd94002bbd63eed3340df1633584e5dbd8f141d7d3d4a7e993037284da",
    "tools/r9700/check_rmsnorm_decode_static.py": "1f50496166b4eed9331214b402222d6c3a6a53e918555e99feee40da4eeb06f7",
    "tools/r9700/test_rmsnorm_decode_routing.cpp": "5b1f01c95c071e97b204caeed74d5fa75161d499487f42eb82796d7a23d27151",
    "tools/r9700/Makefile": "2a7f254d0feb2cedf5b6d65408c340a8f23d17afbd1f2d73213852950334234b",
    "tools/r9700/check_rmsnorm_dflash_rows56_static.py": "cadf49d9d3f963017a41fcdf7e1aca0a09e5198b796d4442a8057b50e3a86d81",
    "tools/r9700/test_check_rmsnorm_dflash_rows56_static.py": "3bd548935130bb36f178a1abbb93e01e03259d46cfc846c06813cf6e001c2736",
}
EXTERNAL = {
    Path("/opt/rocm/bin/hipcc"): "7b95d430bb8c4d4237f9b4935dbd6440e2067fde0979cc69bffb26ebd021464c",
    Path("/opt/rocm/llvm/bin/clang++"): "241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247",
    Path("/opt/rocm/core-10.0/lib/libamdhip64.so.7"): "817aeadfd9f62b68831ad89993163c7f1f470f30595e7e0da5fdc942193142a8",
    Path("/usr/bin/python3"): "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"immutable input differs: {relative}")
    for path, expected in EXTERNAL.items():
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"toolchain/runtime identity differs: {path}")
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
    linked = subprocess.run(["/usr/bin/ldd", str(ROOT / "tools/r9700/build/rmsnorm_dflash_rows56_qual")],
                            check=True, capture_output=True, text=True, env=environment).stdout
    if "not found" in linked or \
            "libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7" not in linked:
        raise SystemExit("executable runtime resolution differs")
    plan = json.loads((PACKAGE / "plan.json").read_text())
    if plan.get("schema") != "ninfer.r9700.rmsnorm-k5120-rows56-plan.v2" or \
            plan["scope"] != {"features": 5120, "rows": [5, 6]} or \
            plan["production_routing"] is not False or plan.get("hardware") != {
                "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
                "pci_vendor_device": "1002:7551", "drm_card": 2,
                "power_profile_before_after": "auto"}:
        raise SystemExit("plan scope differs")
    if plan.get("build", {}).get("binary") != {
            "path": "tools/r9700/build/rmsnorm_dflash_rows56_qual",
            "sha256": EXPECTED["tools/r9700/build/rmsnorm_dflash_rows56_qual"]} or \
            plan.get("build", {}).get("assembly") != {
                "path": "tools/r9700/build/eager_ops.s",
                "sha256": EXPECTED["tools/r9700/build/eager_ops.s"]} or \
            plan.get("build", {}).get("kernel_resources") != {
                "vgprs": 17, "lds_bytes": 32, "occupancy": 16,
                "maximum_workgroup": 256, "wavefront": 32,
                "private_bytes": 0, "scratch_bytes": 0}:
        raise SystemExit("plan build/static identity differs")
    if plan.get("build", {}).get("command") != \
            "make -C tools/r9700 -B rmsnorm-dflash-rows56-build rmsnorm-dflash-rows56-static rmsnorm-dflash-rows56-static-test -j2" or \
            plan.get("toolchain") != {
                "hipcc_sha256": EXTERNAL[Path("/opt/rocm/bin/hipcc")],
                "clang_sha256": EXTERNAL[Path("/opt/rocm/llvm/bin/clang++")],
                "libamdhip64_sha256": EXTERNAL[Path("/opt/rocm/core-10.0/lib/libamdhip64.so.7")]}:
        raise SystemExit("plan command/toolchain identity differs")
    if plan.get("outputs") != ["hardware-before.txt", "hardware-after.txt", "static.stdout",
            "cells.json", "benchmark.stdout", "benchmark.stderr", "benchmark.exit",
            "summary.json", "result.sha256"]:
        raise SystemExit("plan output identity differs")
    print("rows5/6 immutable CPU preflight: PASS")

if __name__ == "__main__": main()
