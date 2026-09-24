"""Host-only strict binding checks against real capped artifacts and sparse wrong-format fixtures."""
import argparse
import json
from pathlib import Path
import struct

from tools.convert.qwen3.common.inventory import TensorSpec, tensor_spec
from tools.convert.qwen3_8_27b_r9700 import convert_fp8_capped as conversion
from tools.convert.qwen3_8_27b_r9700 import compose_fp8_capped_dflash as companion
from tools.r9700.make_sparse_r9700_candidate import build_objects, align_up, PREFIX, MAGIC, PAYLOAD_ALIGNMENT
from tools.ppl.compare_nvfp4 import run


def invalid_fixture(path, recipe, changed, fmt):
    specs=(companion.base_specs()+companion.dflash.TENSOR_SPECS if recipe==companion.WEIGHTS_ID
           else conversion.specs_for(conversion.recipes()[recipe]))
    if changed=='missing-dflash':specs=companion.base_specs()
    specs=tuple((TensorSpec(s.name,s.shape,fmt,conversion.ROW_SCALED_LAYOUT)
                 if fmt==conversion.F8E4M3_ROW_F32S else tensor_spec(s.name,s.shape,fmt))
                if s.name==changed else s for s in specs)
    objects=build_objects(specs,wrong_token_format=False)
    directory=json.dumps(dict(identity=dict(model_id=conversion.q4_inventory.MODEL_ID,
                                             weights_id=recipe),objects=objects),separators=(',',':')).encode()
    offset=align_up(PREFIX.size+len(directory),PAYLOAD_ALIGNMENT)
    with path.open('xb') as f:
        f.write(PREFIX.pack(MAGIC,len(directory)));f.write(directory)
        f.truncate(offset+objects[-1]['offset']+objects[-1]['bytes'])
        ids=next(o for o in objects if o['name']=='text/draft_head_token_ids')
        f.seek(offset+ids['offset']);f.write(struct.pack('<131072I',*range(131072)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--only',choices=['early-attention','all-attention','attention-gdn','selective-cap','selective-cap-dflash'])
    p.add_argument('--artifact',type=Path,help='Explicit companion artifact; otherwise use a sparse host-only valid fixture')
    args=p.parse_args();out=args.out.resolve();fixtures=out/'binder-fixtures';fixtures.mkdir(exist_ok=True)
    if args.only=='selective-cap-dflash':
        recipe=companion.WEIGHTS_ID
        missing=fixtures/'companion-missing-dflash.ninfer'
        wrong=fixtures/'companion-wrong-matrix.ninfer'
        invalid_fixture(missing,recipe,'missing-dflash',conversion.q4_inventory.Q4)
        invalid_fixture(wrong,recipe,'dflash/feature_projection','BF16')
        valid=args.artifact
        if valid is None:
            valid=fixtures/'companion-valid-sparse.ninfer'
            invalid_fixture(valid,recipe,None,None)
        run([args.binary.resolve(),'--fp8-capped-dflash',valid.resolve(),missing,wrong],
            out/'binding-selective-cap-dflash')
        print('selective-cap-dflash: PASS fixed inventory, missing/wrong companion rejection',flush=True)
        return
    for name in ['early-attention','all-attention','attention-gdn','selective-cap']:
        if args.only and args.only!=name:continue
        recipe=f'r9700-q4-fp8-{name}-n16k16-eval'
        selected=sorted(conversion.recipes()[recipe])[0]
        bad_selected=fixtures/f'{name}-selected.ninfer'
        bad_unlisted=fixtures/f'{name}-unlisted.ninfer'
        invalid_fixture(bad_selected,recipe,selected,conversion.q4_inventory.Q4)
        invalid_fixture(bad_unlisted,recipe,'text/layers/0/gdn/value_z',conversion.F8E4M3_ROW_F32S)
        run([args.binary.resolve(),'--fp8-capped',out/'artifacts'/f'{name}.ninfer',bad_selected,bad_unlisted],
            out/f'binding-{name}')
        print(name+': PASS strict real-artifact binding and wrong-format rejection',flush=True)


if __name__=='__main__':main()
