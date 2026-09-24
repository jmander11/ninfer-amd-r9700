"""Summarize explicit trace databases; kernel sums are attribution, not speed evidence."""
import argparse
import sqlite3
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('database', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
db = sqlite3.connect(f'file:{args.database}?mode=ro', uri=True)
launches = db.execute('select graph_exec_id,count(*),min(kernel_dispatch_count),'
                     'max(kernel_dispatch_count) from graph_launches group by graph_exec_id').fetchall()
graphs = []
for graph, count, ms in db.execute('select graph_exec_id,count(*),sum(end-start)/1e6 '
                                  'from kernels group by graph_exec_id'):
    families = [dict(name=name, dispatches=n, kernel_ms=t, fraction=t/ms)
        for name,n,t in db.execute('select name,count(*),sum(end-start)/1e6 from kernels '
            'where graph_exec_id IS ? group by name order by sum(end-start) desc', (graph,))]
    graphs.append(dict(graph=graph, dispatches=count, kernel_ms=ms, families=families))
write_json(args.output, dict(database=str(args.database), timing_use='attribution_only',
                            graph_launches=launches, graphs=graphs))
