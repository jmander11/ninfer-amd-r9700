"""Attach the fixed canonical-Q4/BF16-codebook companion to selective-cap, byte-exact.

This fixed target recipe copies the complete selected base and only dflash/ objects from
the qualified selective-protected tiled-head donor. No quantization or GPU is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorObject
from tools.artifact.container import TensorSpec as StoredTensor, ResourceSpec as StoredResource
from . import convert_fp8_capped as capped
from . import dflash2_q4_inventory as dflash

BASE_ID = 'r9700-q4-fp8-selective-cap-n16k16-eval'
WEIGHTS_ID = 'r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval'
DONOR_ID = dflash.SELECTIVE_WEIGHTS_ID


def base_specs():
    return capped.specs_for(capped.recipes()[BASE_ID])


def stored_spec(obj):
    return (StoredTensor(obj.name, obj.shape, obj.format, obj.layout)
            if isinstance(obj, TensorObject) else StoredResource(obj.name, obj.encoding, obj.bytes))


def payload_hash(artifact, name):
    digest = hashlib.sha256()
    for part in capped.copied_chunks(artifact, name):
        digest.update(part)
    # Release the last mmap view before the owning Artifact closes.
    del part
    return digest.hexdigest()


def validate(path):
    receipt = json.loads(Path(str(path) + '.conversion.json').read_text())
    if receipt['weights_id'] != WEIGHTS_ID:
        raise ValueError('incorrect companion receipt identity')
    with Artifact(path) as artifact:
        capped.validate_inventory(artifact, base_specs() + dflash.TENSOR_SPECS, WEIGHTS_ID)
        if [o.name for o in artifact.objects] != [r['name'] for r in receipt['objects']]:
            raise ValueError('incomplete copied-payload evidence')
        for row in receipt['objects']:
            if payload_hash(artifact, row['name']) != row['sha256']:
                raise ValueError('copied payload differs: ' + row['name'])
    return receipt


def compose(base_path, donor_path, output):
    base_path, donor_path, output = map(Path, (base_path, donor_path, output))
    receipt_path = Path(str(output) + '.conversion.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    # The selected base's conversion receipt binds every represented payload to its recipe.
    capped.validate(base_path)
    report = dict(weights_id=WEIGHTS_ID, recipe=dflash.RECIPE_ID,
                  base=str(base_path.resolve()), donor=str(donor_path.resolve()),
                  base_weights_id=BASE_ID, donor_weights_id=DONOR_ID,
                  source_receipts={},
                  base_objects=1124, companion_objects=66, objects=[])
    for role, path, suffix in [('base', base_path, '.conversion.json'),
                               ('donor', donor_path, '.head-layout.json')]:
        receipt = Path(str(path) + suffix)
        if receipt.is_file():
            report['source_receipts'][role] = dict(
                path=str(receipt.resolve()), sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        capped.validate_inventory(base, base_specs(), BASE_ID)
        capped.validate_inventory(donor, dflash.SELECTIVE_OBJECT_SPECS, DONOR_ID)
        names = tuple(s.name for s in dflash.TENSOR_SPECS)
        companion = tuple(o for o in donor.objects if o.name in names)
        if tuple(o.name for o in companion) != names:
            raise ValueError('donor DFlash object order differs')
        sources = [(base, o) for o in base.objects] + [(donor, o) for o in companion]
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity(dflash.MODEL_ID, WEIGHTS_ID),
                            tuple(stored_spec(o) for _, o in sources)) as writer:
            for source, obj in sources:
                row = dict(name=obj.name, origin='base' if source is base else 'dflash-donor')
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
    parser.add_argument('--base', type=Path)
    parser.add_argument('--donor', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.validate:
        if any((args.base, args.donor, args.out)):
            parser.error('--validate cannot be combined with composition arguments')
        report = validate(args.validate)
    else:
        if not all((args.base, args.donor, args.out)):
            parser.error('--base, --donor and --out are required')
        report = compose(args.base, args.donor, args.out)
    print(json.dumps(report['artifact']))


if __name__ == '__main__':
    main()
