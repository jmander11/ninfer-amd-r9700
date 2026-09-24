"""Build one task-local screen against an explicit saved/current production core."""
import argparse
from pathlib import Path
from tools.ppl.compare_nvfp4 import run
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('label')
p.add_argument('--core',type=Path,default=Path('build-r9700/src/libninfer_r9700_core.a'))
a=p.parse_args();here=Path(__file__).resolve().parent;root=here.parents[2]
assert Path(a.label).name==a.label and not (here/a.label).exists()
run(['/opt/rocm/bin/hipcc','-O3','-std=c++20','--offload-arch=gfx1201',
     '-I'+str(root/'src'),'-I'+str(root/'include'),'-I'+str(root/'tools/r9700'),
     '-I'+str(root/'third_party'),str(a.source.resolve()),'-x','none',str(a.core.resolve()),
     '-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lhipblaslt','-o',str(here/a.label)],
    here/(a.label+'-build'))
