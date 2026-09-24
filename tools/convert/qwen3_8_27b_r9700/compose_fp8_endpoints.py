"""Copy fixed W8 endpoint ablations onto registered FP8 capped evaluation bases.

Embedding retains the donor's row-split layout; output head retains its N16K16
layout. Codes and FP16 scales are copied exactly, with no DFlash objects attached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorObject
from tools.artifact.container import TensorSpec as StoredTensor, ResourceSpec as StoredResource
from tools.convert.qwen3.common.inventory import TensorSpec
from . import convert_fp8_capped as capped
from . import dflash2_q4_inventory as dflash
from .compose_fp8_capped_dflash import payload_hash

AUTHORITY = capped.AUTHORITY.with_name('fp8_endpoint_selection.inc')
EMBEDDING = 'text/token_embedding'
HEAD = 'text/output_head'


def recipes():
    bases = dict(re.findall(r'NINFER_QWEN38_FP8_CAP_RECIPE\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)',
                            capped.AUTHORITY.read_text()))
    records = re.findall(r'NINFER_QWEN38_FP8_ENDPOINT\(\s*\w+\s*,\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*([01])\s*,\s*([01])\s*\)',
                         AUTHORITY.read_text())
    return {identity: (bases[base], bool(int(embed)), bool(int(head)))
            for identity, base, embed, head in records}


def endpoint_names(recipe):
    _, embed, head = recipes()[recipe]
    return {name for name, enabled in ((EMBEDDING, embed), (HEAD, head)) if enabled}


def specs_for(recipe):
    base, _, _ = recipes()[recipe]
    selected = endpoint_names(recipe)
    return tuple(TensorSpec(s.name, s.shape, 'W8G32_F16S',
                           'r9700-w8g32-n16-k16-v1' if s.name == HEAD else 'row-split-k128-v1')
                 if s.name in selected else s
                 for s in capped.specs_for(capped.recipes()[base]))


def validate(path):
    receipt = json.loads(Path(str(path) + '.conversion.json').read_text())
    recipe = receipt['weights_id']
    with Artifact(path) as artifact:
        capped.validate_inventory(artifact, specs_for(recipe), recipe)
        if [o.name for o in artifact.objects] != [r['name'] for r in receipt['objects']]:
            raise ValueError('incomplete endpoint payload evidence')
        for row in receipt['objects']:
            if payload_hash(artifact, row['name']) != row['sha256']:
                raise ValueError('copied payload differs: ' + row['name'])
    return receipt


def compose(recipe, base_path, donor_path, output):
    base_id, _, _ = recipes()[recipe]
    selected = endpoint_names(recipe)
    base_path, donor_path, output = map(Path, (base_path, donor_path, output))
    receipt_path = Path(str(output) + '.conversion.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    expected = specs_for(recipe)
    report = dict(weights_id=recipe, base=str(base_path.resolve()), donor=str(donor_path.resolve()),
                  base_weights_id=base_id, donor_weights_id=dflash.SELECTIVE_WEIGHTS_ID,
                  w8_endpoints=sorted(selected), objects=[])
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        capped.validate_inventory(base, capped.specs_for(capped.recipes()[base_id]), base_id)
        capped.validate_inventory(donor, dflash.SELECTIVE_OBJECT_SPECS, dflash.SELECTIVE_WEIGHTS_ID)
        stored = tuple(StoredTensor(s.name, s.shape, s.format, s.layout)
                       if isinstance(s, TensorSpec) else StoredResource(o.name, o.encoding, o.bytes)
                       for s, o in zip(expected, base.objects, strict=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity(dflash.MODEL_ID, recipe), stored) as writer:
            for obj in base.objects:
                source = donor if obj.name in selected else base
                row = dict(name=obj.name, origin='w8-endpoint-donor' if obj.name in selected else 'base')
                writer.write(obj.name, capped.recorded_chunks(capped.copied_chunks(source, obj.name), row))
                report['objects'].append(row)
    report['artifact'] = dict(path=str(output.resolve()), bytes=output.stat().st_size)
    with receipt_path.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    validate(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=sorted(recipes()))
    parser.add_argument('--base', type=Path)
    parser.add_argument('--donor', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.validate:
        if any((args.recipe, args.base, args.donor, args.out)):
            parser.error('--validate cannot be combined with composition arguments')
        report = validate(args.validate)
    else:
        if not all((args.recipe, args.base, args.donor, args.out)):
            parser.error('--recipe, --base, --donor and --out are required')
        report = compose(args.recipe, args.base, args.donor, args.out)
    print(json.dumps(report['artifact']))


if __name__ == '__main__':
    main()
