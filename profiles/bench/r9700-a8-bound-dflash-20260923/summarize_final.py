"""Close the bounded numerical/decode pass from its explicit retained reports."""
import json
from pathlib import Path
from tools.ppl.compare_nvfp4 import digest, write_json

HERE = Path(__file__).resolve().parent
binary_hash = digest(HERE/'final-bin/bench')
rows = []
for final, baseline in [
    ('final-c1-k5','baseline-c1-k5'), ('final-c1-k4','baseline-c1-k4'),
    ('final-c4-k5','baseline-c4-k5'), ('final-c4-k4','baseline-fixed-c4'),
    ('final-c4-adaptive','baseline-adaptive-c4'),
]:
    selected = json.loads((HERE/final/'summary.json').read_text())
    control = json.loads((HERE/baseline/'summary.json').read_text())
    receipt = json.loads((HERE/final/'receipt.json').read_text())
    assert receipt['exit_code'] == 0 and receipt['executable_sha256'] == binary_hash
    assert selected['all_repetitions_exact_ordinary'] and control['all_repetitions_exact_ordinary']
    assert selected['concurrency'] == control['concurrency']
    assert selected['max_draft'] == control['max_draft']
    assert selected['adaptive'] == control['adaptive']
    old, new = control['aggregate_decode_tok_s'], selected['aggregate_decode_tok_s']
    rows.append(dict(cell=final, baseline=baseline, aggregate_before=old, aggregate_after=new,
                     per_request_after=new/selected['concurrency'], speedup_percent=100*(new/old-1),
                     all_six_repetitions_exact_ordinary=True,
                     final_repetition_rates=selected['repetition_rates'],
                     final_request_lane_rounds_per_draft=selected['request_lane_rounds_per_draft']))

numeric = json.loads((HERE/'final-draft-qualification.json').read_text())
assert numeric['status'] == 'qualified' and numeric['scope'] == 'draft_only'
assert {(c['rows'],c['columns'],c['tokens']) for c in numeric['cells']} == {
    (5120,k,t) for k in [17408,25600] for t in [5,6]}
assert json.loads((HERE/'final-code-object.json').read_text())['exact_qualified_code_object']
write_json(HERE/'final-summary.json', dict(status='complete', rows=rows,
    final_benchmark_sha256=binary_hash, final_public_draft_cells=4,
    production_precision_changed=False, model_weights_changed=False,
    prefill_chunk=2048, prefill_work='paused',
    adaptive_policy='unchanged; measured tail limitation, not a correctness bug',
    hardware_ceiling_claim=False))
print(json.dumps(rows, indent=2))
