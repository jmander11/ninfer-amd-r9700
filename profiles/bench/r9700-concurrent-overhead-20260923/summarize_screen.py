"""Summarize exact-matched operator pairs; distinguish retained from new routes."""
import json
import statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json
here=Path(__file__).resolve().parent
data=json.loads((here/'projection-candidate.json').read_text())
rows=[]
for cell in data['cells']:
    control=statistics.median(x['control_ms'] for x in cell['samples'])
    selected=statistics.median(x['selected_ms'] for x in cell['samples'])
    retained=cell['rows']==5120 and cell['columns']==6144 and cell['tokens'] in [12,18,24]
    rows.append(dict(rows=cell['rows'],columns=cell['columns'],tokens=cell['tokens'],
        generic_ms=control,selected_ms=selected,saving_fraction=1-selected/control,
        new_route=not retained,comparison='generic, not previous production' if retained else 'previous production generic'))
write_json(here/'projection-speed-summary.json',dict(cells=rows,
           new_route_count=sum(r['new_route'] for r in rows)))
for shape in sorted({(r['rows'],r['columns']) for r in rows}):
    selected=[r for r in rows if (r['rows'],r['columns'])==shape and r['new_route']]
    print(shape,'savings %',[(r['tokens'],round(100*r['saving_fraction'],2)) for r in selected])
if any(r['new_route'] and r['saving_fraction'] < 0.02 for r in rows):
    raise RuntimeError('a new route lacks a clear operator win; retain report and review before whole timing')
