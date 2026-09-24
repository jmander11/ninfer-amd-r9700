"""Create-only conversion of the fixed Q4/FP8 capped evaluation recipes.

Copy represented Q4/FP8 bytes from validated local donors. Only protections not
present as FP8 in either donor are encoded from the original BF16 checkpoint.
The target's fixed selection record owns the exact per-recipe matrix inventory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ArtifactWriter, TensorObject,
    TensorSpec as StoredTensor, ResourceSpec as StoredResource,
)
from tools.convert.qwen3.common.inventory import TensorSpec
from . import q4_inventory, fp8_hybrid_inventory, selective_protected_inventory
from .e4m3_inventory import F8E4M3_ROW_F32S, ROW_SCALED_LAYOUT

AUTHORITY = Path(__file__).resolve().parents[3] / 'src/targets/qwen3_8_27b/impl/load/fp8_capped_selection.inc'


def recipes():
    text = AUTHORITY.read_text()
    identities = dict(re.findall(r'NINFER_QWEN38_FP8_CAP_RECIPE\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', text))
    selected = {key: set() for key in identities}
    for key, name in re.findall(r'NINFER_QWEN38_FP8_CAP_MATRIX\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', text):
        if name in selected[key]:
            raise ValueError(f'duplicate selected matrix: {key}/{name}')
        selected[key].add(name)
    base = {s.name: s for s in q4_inventory.TENSOR_SPECS}
    for key, names in selected.items():
        if not names or any(n not in base or base[n].format != q4_inventory.Q4 for n in names):
            raise ValueError(f'capped recipe must replace existing Q4 matrices: {key}')
    return {identities[key]: names for key,names in selected.items()}


def specs_for(selected):
    return tuple(TensorSpec(s.name, s.shape, F8E4M3_ROW_F32S, ROW_SCALED_LAYOUT)
                 if s.name in selected else s for s in q4_inventory.OBJECT_SPECS)


def validate_inventory(artifact, expected, weights_id):
    if artifact.identity != ArtifactIdentity(q4_inventory.MODEL_ID, weights_id):
        raise ValueError(f'incorrect donor/output identity: {artifact.identity}')
    if len(artifact.objects) != len(expected):
        raise ValueError('incorrect object inventory length')
    for obj,spec in zip(artifact.objects, expected, strict=True):
        if obj.name != spec.name:
            raise ValueError('incorrect object order')
        if isinstance(spec, TensorSpec):
            if not isinstance(obj,TensorObject) or (obj.shape,obj.format,obj.layout) != (spec.shape,spec.format,spec.layout):
                raise ValueError(f'incorrect tensor contract: {spec.name}')
        elif isinstance(obj,TensorObject) or obj.encoding != spec.encoding:
            raise ValueError(f'incorrect resource contract: {spec.name}')


def copied_chunks(artifact, name):
    with artifact.payload(name) as payload:
        for begin in range(0,len(payload),8 << 20):
            yield payload[begin:begin+(8 << 20)]


def recorded_chunks(chunks, record):
    h = hashlib.sha256()
    for part in chunks:
        h.update(part)
        yield part
    record['sha256'] = h.hexdigest()


def convert(args):
    selected = recipes()[args.recipe]
    expected = specs_for(selected)
    receipt_path = Path(str(args.out)+'.conversion.json')
    if args.out.exists() or receipt_path.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report = dict(recipe=args.recipe, weight_recipe_selected=False,
                  fp8_matrices=sorted(selected), objects=[],
                  authority_sha256=hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
                  donors={}, source_model=str(args.model.resolve()) if args.model else None)
    with ExitStack() as stack:
        base = stack.enter_context(Artifact(args.base))
        capped = four = selective = None
        inputs = [('base',base,q4_inventory,args.base)]
        if args.capped_donor:
            capped = stack.enter_context(Artifact(args.capped_donor))
            capped_names = recipes().get(capped.identity.weights_id)
            if capped_names is None or not selected <= capped_names:
                raise ValueError('capped donor must contain every selected FP8 matrix')
            validate_inventory(capped,specs_for(capped_names),capped.identity.weights_id)
            report['donors']['capped'] = dict(path=str(args.capped_donor.resolve()),
                bytes=args.capped_donor.stat().st_size,weights_id=capped.identity.weights_id)
        else:
            four = stack.enter_context(Artifact(args.four_role))
            selective = stack.enter_context(Artifact(args.selective))
            inputs += [('four_role',four,fp8_hybrid_inventory,args.four_role),
                       ('selective',selective,selective_protected_inventory,args.selective)]
        for tag,artifact,inv,path in inputs:
            validate_inventory(artifact,inv.OBJECT_SPECS,inv.WEIGHTS_ID)
            report['donors'][tag] = dict(path=str(path.resolve()),bytes=path.stat().st_size,
                                        weights_id=inv.WEIGHTS_ID)
        four_names = fp8_hybrid_inventory.SELECTED_MATRIX_NAMES
        selective_names = selective_protected_inventory.FP8_NAMES
        source_names = set() if capped else selected-four_names-selective_names
        reader = None
        if source_names:
            import torch
            from tools.convert.common.safetensors import ShardReader
            from . import source_recipe, source
            from .e4m3_rowwise import encode_e4m3_rowwise_chunks
            torch.set_num_threads(4)
            source.validate_config(json.loads((args.model/'config.json').read_text()))
            metadata = source_recipe.preflight_sources(args.model)
            if metadata.source_shard_count != 18 or metadata.source_dtype_counts != {'BF16':1199}:
                raise ValueError('requires complete original BF16 source')
            reader = stack.enter_context(ShardReader(args.model))
        specs = {s.name:s for s in expected if isinstance(s,TensorSpec)}
        stored = tuple(StoredTensor(o.name,specs[o.name].shape,specs[o.name].format,specs[o.name].layout)
                       if isinstance(o,TensorObject) else StoredResource(o.name,o.encoding,o.bytes)
                       for o in base.objects)
        with ArtifactWriter(args.out,ArtifactIdentity(q4_inventory.MODEL_ID,args.recipe),stored) as writer:
            for obj in base.objects:
                name=obj.name
                record=dict(name=name)
                if name in source_names:
                    tensor=source_recipe.materialize_recipe(source_recipe.RECIPES_BY_NAME[name],reader)
                    record['origin']='original-bf16-source'
                    record['source_sha256']=hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B')).hexdigest()
                    writer.write(name,recorded_chunks(encode_e4m3_rowwise_chunks(tensor),record))
                    del tensor
                else:
                    tag,donor = (('capped',capped) if name in selected and capped else
                                 ('four_role',four) if name in selected and name in four_names else
                                 ('selective',selective) if name in selected else ('base',base))
                    record['origin']=tag+'-copy-exact'
                    writer.write(name,recorded_chunks(copied_chunks(donor,name),record))
                report['objects'].append(record)
    report['artifact']=dict(path=str(args.out.resolve()),bytes=args.out.stat().st_size)
    with receipt_path.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    validate(args.out)
    print(json.dumps(report['artifact']),flush=True)


def validate(path):
    report=json.loads(Path(str(path)+'.conversion.json').read_text())
    selected=recipes()[report['recipe']]
    if report['fp8_matrices'] != sorted(selected):
        raise ValueError('receipt recipe differs')
    with Artifact(path) as artifact:
        validate_inventory(artifact,specs_for(selected),report['recipe'])
        if [r['name'] for r in report['objects']] != [o.name for o in artifact.objects]:
            raise ValueError('incomplete payload evidence')
        for row in report['objects']:
            h=hashlib.sha256()
            for part in copied_chunks(artifact,row['name']):h.update(part)
            del part  # Drop the last mmap slice before closing the artifact.
            if h.hexdigest() != row['sha256']:
                raise ValueError(f"payload differs: {row['name']}")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',choices=sorted(recipes()))
    p.add_argument('--base',type=Path)
    p.add_argument('--four-role',type=Path)
    p.add_argument('--selective',type=Path)
    p.add_argument('--capped-donor',type=Path,
                   help='Copy selected FP8 tensors from this registered capped base; no source encoding')
    p.add_argument('--model',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--validate',type=Path)
    args=p.parse_args()
    if args.validate:
        validate(args.validate);print('PASS: fixed inventory and every payload digest')
    else:
        required = ('recipe','base','out') + (() if args.capped_donor else ('four_role','selective','model'))
        for key in required:
            if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' is required')
        if args.capped_donor and any((args.four_role,args.selective,args.model)):
            p.error('--capped-donor replaces --four-role, --selective and --model')
        convert(args)


if __name__ == '__main__':main()
