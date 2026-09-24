"""Isolate HEAD adaptive policy without changing the working tree or core kernels."""
import shlex
import subprocess
from pathlib import Path
from tools.ppl.compare_nvfp4 import run

here=Path(__file__).resolve().parent
root=here.parents[2]
build=root/'build-r9700'
for name in ['adaptive_draft.h', 'program_impl.h', 'decision_trace.h']:
    relative='targets/qwen3/impl/runtime/'+name
    source=subprocess.check_output(['git','-C',root,'show','c63c893f:src/'+relative])
    destination=here/'control-src'/relative
    if destination.exists():
        assert destination.read_bytes()==source
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source)
lines=subprocess.check_output(['ninja','-C',build,'-t','commands','ninfer_r9700_qwen3_8_27b_runtime'],text=True).splitlines()
line=next(x for x in lines if ' -c ' in x and x.endswith('/impl/runtime.hip'))
command=shlex.split(line)
for flag in ['-MT','-MF']:
    i=command.index(flag);del command[i:i+2]
command.remove('-MD')
command[command.index('-o')+1]=str(here/'control-runtime.o')
command.insert(1,'-I'+str(here/'control-src'))
run(command,here/'control-runtime-build')
line=subprocess.check_output(['ninja','-C',build,'-t','commands','ninfer_bench'],text=True).splitlines()[-1]
command=shlex.split(line.split('&&')[1])
command=[x for x in command if not x.endswith('.cpp.o')]
command[command.index('-o')+1]=str(here/'tail-check-control')
command=[str(build/x) if x.endswith('.a') and not x.startswith('/') else x for x in command]
command.insert(command.index(str(build/'src/libninfer_r9700_qwen3_8_27b_runtime.a')),
               str(here/'control-runtime.o'))
command[1:1]=['-std=c++20','-I'+str(root/'include'),str(here/'tail_check.cpp')]
run(command,here/'control-tail-link')
