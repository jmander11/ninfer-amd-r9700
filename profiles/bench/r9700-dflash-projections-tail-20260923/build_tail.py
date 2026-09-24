"""Link the small Engine regression against the exact benchmark library closure."""
import shlex
import subprocess
import sys
from pathlib import Path
from tools.ppl.compare_nvfp4 import run

here = Path(__file__).resolve().parent
root = here.parents[2]
build = root/'build-r9700'
label = sys.argv[1]
assert Path(label).name == label
line = subprocess.check_output(['ninja', '-C', build, '-t', 'commands', 'ninfer_bench'], text=True).splitlines()[-1]
command = shlex.split(line.split('&&')[1])
command = [arg for arg in command if not arg.endswith('.cpp.o')]
command[command.index('-o')+1] = str(here/label)
command = [str(build/arg) if arg.endswith('.a') and not arg.startswith('/') else arg for arg in command]
command[1:1] = ['-std=c++20', '-I'+str(root/'include'), str(here/'tail_check.cpp')]
assert not (here/label).exists()
run(command, here/(label+'-build'))
