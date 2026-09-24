"""Build the same bounded complete-attention screen against the current core."""
import argparse
from pathlib import Path
from tools.ppl.compare_nvfp4 import run

here = Path(__file__).resolve().parent
root = here.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("label", help="Fresh executable basename, e.g. attention-control")
args = parser.parse_args()
if Path(args.label).name != args.label or args.label in (".", ".."):
    parser.error("label must be a basename")
output = here / args.label
if output.exists():
    parser.error("output already exists")
command = ["/opt/rocm/bin/hipcc", "-O3", "-std=c++20", "--offload-arch=gfx1201",
           "-I" + str(root / "src"), "-I" + str(root / "include"),
           "-I" + str(root / "tools/r9700"),
           "-I" + str(root / "third_party"),
           str(here / "attention_screen.hip"), "-x", "none",
           str(root / "build-r9700/src/libninfer_r9700_core.a"),
           "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib", "-lhipblaslt",
           "-o", str(output)]
run(command, here / (args.label + "-build"))
