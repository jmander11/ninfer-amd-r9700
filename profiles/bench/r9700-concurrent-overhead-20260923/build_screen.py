"""Compile a standalone public-Op screen against the current production core."""
import sys
from pathlib import Path
from tools.ppl.compare_nvfp4 import run
here=Path(__file__).resolve().parent
root=here.parents[2]
label=sys.argv[1]
assert Path(label).name==label
cmd=['/opt/rocm/bin/hipcc','-O3','-std=c++20','--offload-arch=gfx1201',
     '-I'+str(root/'src'),'-I'+str(root/'include'),'-I'+str(root/'tools/r9700'),
     str(here/'projection_screen.hip'),'-x','none',str(root/'build-r9700/src/libninfer_r9700_core.a'),
     '-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lhipblaslt','-o',str(here/label)]
assert not (here/label).exists()
run(cmd,here/(label+'-build'))
