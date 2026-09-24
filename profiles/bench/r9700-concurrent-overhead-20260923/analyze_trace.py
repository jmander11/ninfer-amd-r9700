"""Offline graph-service and gap attribution, never a performance-admission timing."""
import argparse
import bisect
import collections
import json
import sqlite3
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('database',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
c=sqlite3.connect(f'file:{a.database.resolve()}?mode=ro',uri=True)
c.row_factory=sqlite3.Row
kernels=list(c.execute('select * from kernels order by start'))
copies=list(c.execute('select * from memory_copies order by start'))
regions=list(c.execute("select * from regions where category='HIP_RUNTIME_API_EXT' order by start"))
def union(intervals):
    total=0; end=-1
    for x,y in sorted(intervals):
        total+=max(0,y-max(x,end)); end=max(end,y)
    return total
def clipped(rows,start,end):
    return [(max(start,r['start']),min(end,r['end'])) for r in rows
            if r['start']<end and r['end']>start]
groups={}
for r in kernels:
    if r['graph_exec_id'] and r['graph_node_id']==0:
        groups.setdefault(r['graph_exec_id'],[]).append(r['start'])
rounds=collections.defaultdict(list)
for r in kernels:
    g=r['graph_exec_id']
    if g in groups:
        i=bisect.bisect_right(groups[g],r['start'])-1
        assert i>=0
        rounds[g,i].append(r)
rows=[]
for (g,i),ks in rounds.items():
    start=min(k['start'] for k in ks); end=max(k['end'] for k in ks)
    service=union((k['start'],k['end']) for k in ks)
    busy=union([(k['start'],k['end']) for k in ks]+clipped(copies,start,end))
    rows.append(dict(graph=g,round=i,start=start,end=end,kernels=len(ks),
                     span_ms=(end-start)/1e6,kernel_busy_ms=service/1e6,
                     uncovered_in_graph_ms=(end-start-busy)/1e6))
rows.sort(key=lambda r:r['start'])
gaps=[]
for previous,nxt in zip(rows,rows[1:]):
    start=previous['end']; end=nxt['start']
    assert end>=start
    busy=union(clipped(kernels+copies,start,end))
    apis=collections.defaultdict(list)
    for r in regions:
        if r['start']<end and r['end']>start:
            apis[r['name']].append((max(start,r['start']),min(end,r['end'])))
    gaps.append(dict(from_graph=previous['graph'],to_graph=nxt['graph'],gap_ms=(end-start)/1e6,
                     device_busy_ms=busy/1e6,uncovered_ms=(end-start-busy)/1e6,
                     clipped_api_ms={n:union(v)/1e6 for n,v in apis.items()}))
families=[]; generic=[]
for g in groups:
    ks=[k for k in kernels if k['graph_exec_id']==g]
    total=sum(k['duration'] for k in ks)
    buckets=collections.defaultdict(list)
    for k in ks:buckets[k['name']].append(k)
    fam=[dict(name=n,calls=len(v),kernel_ms=sum(k['duration'] for k in v)/1e6,
              fraction=sum(k['duration'] for k in v)/total) for n,v in buckets.items()]
    families.append(dict(graph=g,rounds=len(groups[g]),kernel_ms=total/1e6,
                         families=sorted(fam,key=lambda x:-x['kernel_ms'])))
    buckets=collections.defaultdict(list)
    for k in ks:
        if 'a8q4g64_linear_wmma32_kernel' in k['name']:
            buckets[k['graph_node_id'],k['grid_x'],k['grid_y']].append(k)
    generic += [dict(graph=g,node=n,grid_x=x,grid_y=y,calls=len(v),
                     mean_us=sum(k['duration'] for k in v)/len(v)/1000)
                for (n,x,y),v in buckets.items()]
out=dict(database=str(a.database),timing_use='attribution_only',rounds=rows,gaps=gaps,
         families=families,generic_nodes=generic,
         graph_span_ms=sum(r['span_ms'] for r in rows),
         graph_kernel_busy_ms=sum(r['kernel_busy_ms'] for r in rows),
         graph_uncovered_ms=sum(r['uncovered_in_graph_ms'] for r in rows),
         inter_round_gap_ms=sum(g['gap_ms'] for g in gaps),
         inter_round_uncovered_ms=sum(g['uncovered_ms'] for g in gaps))
write_json(a.output,out)
print(json.dumps({k:v for k,v in out.items() if k not in ['rounds','gaps','families','generic_nodes']},indent=2))
