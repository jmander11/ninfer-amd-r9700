"""Serial real-Engine greedy/token-limit/context and eager transition checks."""
import json
import os
import sys
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

here = Path(__file__).resolve().parent
root = here.parents[2]
binary = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir()
stride = sys.argv[3] if len(sys.argv)>3 else '17'
artifact = Path('/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval.ninfer')
trace = out/'tail-decisions.json'
reports = {}
for mode in ['ordinary', 'graph', 'eager']:
    if mode == 'eager':
        os.environ['NINFER_DFLASH_DECISION_TRACE_OUT'] = str(trace)
    run([binary, artifact,
         root/'profiles/bench/r9700-compact-mixed-delivery-20260923/code.ids', mode,
         out/f'tail-{mode}.json', stride], out/f'tail-{mode}-run')
    reports[mode] = json.loads((out/f'tail-{mode}.json').read_text())['cases']
    assert reports[mode] == reports['ordinary'], f'{mode} differs from ordinary'
    print(mode, 'exact public tokens and output limits', flush=True)
events = json.loads(trace.read_text())['events']
rows = [e for e in events if e['kind'] == 'dflash']
padded = [e for e in rows if e['physical_verify_width']-1 > e['proposal_extent']]
pending_six_at_k4 = [e for e in rows if e['physical_verify_width'] == 5 and
                    e['base_frontier']-e['context_frontier'] == 6]
zero = [e for e in rows if e['proposal_extent'] == 0]
assert padded, 'no padded physical route observed'
assert zero, 'no target-only fallback observed'
for e in rows:
    assert 1 <= len(e['licensed_tokens']) <= e['proposal_extent']+1
    assert e['next_frontier'] == e['base_frontier']+len(e['licensed_tokens'])
write_json(out/'tail-check-summary.json', dict(all_modes_exact_ordinary=True,
    cases_per_mode=len(reports['ordinary']), eager_round_rows=len(rows),
    padded_rows=len(padded), zero_extent_rows=len(zero),
    pending_six_at_k4_rows=len(pending_six_at_k4),
    timing_eligible=False))
print('transition diagnostics:', len(padded), 'padded;', len(zero), 'target-only;',
      len(pending_six_at_k4), 'K4 with pending six rows', flush=True)
