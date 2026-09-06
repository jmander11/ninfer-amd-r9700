#!/usr/bin/env python3
"""Static admission for the product-source N16/K16 scalar-base qualifier."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bench.extract_embedded_code_object import extract

SYMBOL = "a8q4g64_linear_prefill_cta_scalar_base_qualification_kernel"
BASE_SYMBOL = "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8q4g64_linear_prefill_cta_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj"
MAX_VOFFSETS = {
    "activation_code": 71_303_164,
    "weight_code": 89_128_952,
    "activation_scale": 4_456_446,
    "weight_scale": 5_570_558,
}
MAX_FINAL_BYTES = {
    "activation_code": 71_303_167,
    "weight_code": 89_128_959,
    "activation_scale": 4_456_447,
    "weight_scale": 5_570_559,
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def kernel(text: str) -> str:
    match = re.search(
        rf"(?ms)^{SYMBOL}:.*?^\s*\.size\s+{SYMBOL},",
        text,
    )
    need(match is not None, "candidate symbol missing")
    return match.group(0)


def listing_kernel(text: str) -> str:
    match = re.search(
        rf"(?ms)^{SYMBOL}:.*?(?=^\s*\.protected\s+[^;]+; -- Begin function|\Z)", text
    )
    need(match is not None, "candidate listing section missing")
    return match.group(0)


def object_kernel(text: str, symbol: str) -> str:
    match=re.search(rf"(?ms)^[0-9a-f]+ <{re.escape(symbol)}>:\n(.*?)(?=^[0-9a-f]+ <[^>]+>:\n|\Z)",text)
    need(match is not None, f"loaded object symbol {symbol}")
    return match.group(1)


def opcode_count(text: str, opcode: str) -> int:
    return sum(re.search(rf"^\s*{re.escape(opcode)}(?:\s|$)",line) is not None for line in text.splitlines())


def check_loaded_object(candidate: str, baseline: str, notes: str) -> None:
    cand=object_kernel(candidate,SYMBOL);base=object_kernel(baseline,BASE_SYMBOL)
    symbol_address = int(re.search(
        rf"(?m)^([0-9a-f]+) <{re.escape(SYMBOL)}>:$", candidate
    ).group(1), 16)
    blocks = [block for block in re.findall(r"(?ms)^  - \.args:.*?(?=^  - \.args:|\Z)", notes)
              if re.search(rf"(?m)^    \.name:\s+{re.escape(SYMBOL)}\s*$", block)]
    need(len(blocks) == 1, "unique candidate metadata block")
    metadata = blocks[0]
    pointer_args=[]
    for match in re.finditer(r"(?m)^\s+- \.address_space:\s+(\w+)\n\s+\.offset:\s+(\d+)\n\s+\.size:\s+(\d+)\n\s+\.value_kind:\s+(\w+)",metadata):pointer_args.append((match.group(1),int(match.group(2)),int(match.group(3)),match.group(4)))
    value_args=[(int(a),int(b),c) for a,b,c in re.findall(r"(?m)^\s+- \.offset:\s+(\d+)\n\s+\.size:\s+(\d+)\n\s+\.value_kind:\s+(\w+)",metadata)]
    need(pointer_args==[("global",offset,8,"global_buffer") for offset in range(0,56,8)],"exact seven pointer args")
    need(value_args==[(offset,4,"by_value") for offset in range(56,72,4)],"exact four scalar args")
    for pattern,label in ((rf"\.name:\s+{SYMBOL}\s*$","metadata name"),(rf"\.symbol:\s+{SYMBOL}\.kd\s*$","metadata symbol"),(r"\.kernarg_segment_align:\s+8\b","ABI alignment"),(r"\.kernarg_segment_size:\s+72\b","72-byte ABI"),(r"\.group_segment_fixed_size:\s+17152\b","LDS metadata"),(r"\.private_segment_fixed_size:\s+0\b","private metadata"),(r"\.max_flat_workgroup_size:\s+512\b","workgroup metadata"),(r"\.wavefront_size:\s+32\b","wave32 metadata"),(r"\.workgroup_processor_mode:\s+1\b","WGP metadata"),(r"\.uniform_work_group_size:\s+1\b","uniform workgroup"),(r"\.uses_dynamic_stack:\s+false\b","dynamic stack"),(r"\.sgpr_spill_count:\s+0\b","SGPR spill metadata"),(r"\.vgpr_spill_count:\s+0\b","VGPR spill metadata")):
        need(re.search(pattern,metadata,re.MULTILINE) is not None,label)
    need(int(re.search(r"\.vgpr_count:\s+(\d+)", metadata).group(1)) <= 92,
         "loaded-object VGPR ceiling")
    # The executable object, not the parallel compiler listing, owns these loads.
    loaded=[line.strip().split("//",1)[0].rstrip() for line in cand.splitlines() if "global_load_" in line]
    expected=[("global_load_b32","s[4:5]"),("global_load_b32","s[6:7]"),("global_load_b64","s[16:17]"),("global_load_d16_b16","s[8:9]"),("global_load_d16_b16","s[18:19]")]*2
    need(len(loaded)==10,"loaded-object load inventory")
    need([(line.split()[0],re.search(r"s\[\d+:\d+\]",line).group()) for line in loaded]==expected,"loaded-object load role/order/base")
    for line in loaded:
        need(re.search(r"global_load_(?:b32|b64|d16_b16)\s+v(?:\[\d+:\d+\]|\d+),\s+v\d+,\s+s\[\d+:\d+\]\s*$",line) is not None,"loaded-object scalar base/one VGPR/default policy/offset0")
        opcode=line.split()[0];mc_name={"global_load_b32":"GLOBAL_LOAD_DWORD_SADDR_gfx12","global_load_b64":"GLOBAL_LOAD_DWORDX2_SADDR_gfx12","global_load_d16_b16":"GLOBAL_LOAD_SHORT_D16_SADDR_gfx12"}[opcode]
        result=subprocess.run(["/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc","-triple=amdgcn-amd-amdhsa","-mcpu=gfx1201","-show-inst"],input=line+"\n",text=True,capture_output=True,check=False)
        need(result.returncode==0 and mc_name in result.stdout,f"loaded-object MC class {opcode}")
    ci=cand.splitlines();load_positions=[i for i,x in enumerate(ci) if "global_load_" in x];steady=load_positions[5:8];first=next(i for i,x in enumerate(ci) if "v_wmma_i32_16x16x32_iu4" in x);last=max(i for i,x in enumerate(ci) if "v_wmma_i32_16x16x32_iu4" in x);drain=next(i for i,x in enumerate(ci[min(steady):],min(steady)) if "s_wait_loadcnt" in x)
    need(max(steady)<first<=last<drain,"loaded-object complete successor triplet overlap")
    for position in (load_positions[3], load_positions[4], load_positions[8], load_positions[9]):
        need(any("s_cbranch_execz" in line for line in ci[max(0, position - 8):position]),
             "loaded-object scale-load exec guard")
    backward = False
    for line in ci:
        if "s_cbranch_" not in line:
            continue
        match = re.search(
            rf"//\s*([0-9A-Fa-f]+):.*<{re.escape(SYMBOL)}\+0x([0-9A-Fa-f]+)>",
            line,
        )
        if match and int(match.group(2), 16) < int(match.group(1), 16) - symbol_address:
            backward = True
    need(backward, "loaded-object backward loop edge")
    wait1=next(i for i,x in enumerate(ci[last:],last) if "s_wait_loadcnt 0x1" in x);publish_a=next(i for i,x in enumerate(ci[wait1:],wait1) if "ds_store_2addr_stride64_b32" in x);wait0=next(i for i,x in enumerate(ci[publish_a:],publish_a) if "s_wait_loadcnt 0x0" in x);publish_w=next(i for i,x in enumerate(ci[wait0:],wait0) if "ds_store_2addr_stride64_b32" in x)
    need(last<wait1<publish_a<wait0<publish_w,"loaded-object wait/publish chronology")
    loaded_iu4 = [line for line in ci if "v_wmma_i32_16x16x32_iu4" in line]
    need(len(loaded_iu4) == 8, "loaded-object native IU4 inventory")
    need(sum("neg_lo:[0,1,0]" in line for line in loaded_iu4) == 4,
         "loaded-object unsigned A-low/signed W")
    need(sum("neg_lo:[1,1,0]" in line for line in loaded_iu4) == 4,
         "loaded-object signed A-high/signed W")
    exact_counts=("s_wait_loadcnt","s_wait_dscnt","s_barrier_signal","s_barrier_wait","s_cbranch_vccnz","s_cbranch_vccz","s_cbranch_scc0","s_cbranch_scc1","s_cbranch_execz","s_cbranch_execnz","ds_load_2addr_b32","ds_load_2addr_stride64_b32","ds_load_u16_d16","ds_store_2addr_stride64_b32","ds_store_b16","v_wmma_i32_16x16x32_iu4","v_cvt_f32_i32_e32","v_add3_u32","v_fmac_f32_e32","v_fma_mix_f32","v_dual_fmac_f32","global_store_d16_hi_b16")
    for opcode in exact_counts:need(opcode_count(cand,opcode)==opcode_count(base,opcode),f"production topology {opcode}")
    need(opcode_count(cand,"s_barrier_signal")==opcode_count(cand,"s_barrier_wait")==2,"barrier pairing")
    signals=[i for i,x in enumerate(ci) if "s_barrier_signal" in x];waits=[i for i,x in enumerate(ci) if "s_barrier_wait" in x]
    need(signals[0]<waits[0]<signals[1]<waits[1],"barrier chronology")
    first_convert=next(i for i,x in enumerate(ci) if "v_cvt_f32_i32_e32" in x);first_accumulate=next(i for i,x in enumerate(ci) if "v_fmac_f32_e32" in x or "v_dual_fmac_f32" in x)
    need(first_convert<first_accumulate,"integer recombine/conversion before accumulation")


def integer(text: str, pattern: str, label: str) -> int:
    match = re.search(pattern, text)
    need(match is not None, label)
    return int(match.group(1))


def check(source: str, assembly: str, candidate_disassembly: str|None=None, baseline_disassembly: str|None=None, notes: str|None=None) -> dict[str, int | str]:
    need(assembly.count('.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"') == 1,
         "exact gfx1201 assembly target")
    body = kernel(assembly)
    listing = listing_kernel(assembly)
    vgpr = integer(assembly, rf"\.L{SYMBOL}\.num_vgpr,\s*(\d+)", "VGPR")
    need(vgpr <= 92, "logical VGPR exceeds 92")
    need(integer(listing, r"\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS") == 17152, "LDS")
    need(integer(listing, r"\.amdhsa_private_segment_fixed_size\s+(\d+)", "private") == 0, "private")
    need(integer(listing, r"\.amdhsa_kernarg_size\s+(\d+)", "kernarg") == 72, "kernarg")
    need(integer(listing, r"; ScratchSize:\s*(\d+)", "scratch") == 0, "scratch")
    need(integer(listing, r"; Occupancy:\s*(\d+)", "occupancy") == 16, "occupancy")
    need(integer(listing, r"\.sgpr_spill_count:\s*(\d+)", "SGPR spills") == 0, "SGPR spills")
    need(integer(listing, r"\.vgpr_spill_count:\s*(\d+)", "VGPR spills") == 0, "VGPR spills")

    loads = [line.strip() for line in body.splitlines() if "global_load_" in line]
    expected = {
        ("global_load_b32", "s[4:5]"): 2,
        ("global_load_b32", "s[6:7]"): 2,
        ("global_load_b64", "s[16:17]"): 2,
        ("global_load_d16_b16", "s[8:9]"): 2,
        ("global_load_d16_b16", "s[18:19]"): 2,
    }
    need(len(loads) == 10, "unexpected global-load inventory")
    for (opcode, base), count in expected.items():
        selected = [line for line in loads if opcode in line and base in line]
        need(len(selected) == count, f"{opcode}/{base} count")
        need(all(re.search(rf"{opcode}\s+v(?:\[\d+:\d+\]|\d+),\s+v\d+,\s+{re.escape(base)}\s*$", line) for line in selected), "single-VGPR voffset/default policy/zero immediate")
        mc_name={"global_load_b32":"GLOBAL_LOAD_DWORD_SADDR_gfx12","global_load_b64":"GLOBAL_LOAD_DWORDX2_SADDR_gfx12","global_load_d16_b16":"GLOBAL_LOAD_SHORT_D16_SADDR_gfx12"}[opcode]
        for line in selected:
            instruction=line.split("//",1)[0].strip()
            result=subprocess.run(["/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc","-triple=amdgcn-amd-amdhsa","-mcpu=gfx1201","-show-inst"],input=instruction+"\n",text=True,capture_output=True,check=False)
            need(result.returncode==0 and mc_name in result.stdout, f"MC scalar-base class {opcode}")
    ordered = [(next(k for k in ("b32", "b64", "d16") if k in line), re.search(r"s\[\d+:\d+\]", line).group()) for line in loads]
    need(ordered[:5] == [("b32","s[4:5]"),("b32","s[6:7]"),("b64","s[16:17]"),("d16","s[8:9]"),("d16","s[18:19]")], "prologue load order")
    need(ordered[5:] == ordered[:5], "steady load role/order")
    lines=body.splitlines()
    load_positions = [i for i, line in enumerate(lines) if "global_load_" in line]
    need(len(load_positions) == 10, "compiler-listing global-load inventory")
    steady_start = load_positions[5]
    first_wmma=next(i for i,line in enumerate(lines) if "v_wmma_i32_16x16x32_iu4" in line)
    last_wmma=max(i for i,line in enumerate(lines) if "v_wmma_i32_16x16x32_iu4" in line)
    first_drain=next(i for i,line in enumerate(lines[steady_start:],steady_start) if "s_wait_loadcnt" in line)
    need(steady_start < first_wmma <= last_wmma < first_drain, "successor load overlap/drain chronology")
    for position in (load_positions[3], load_positions[4], load_positions[8], load_positions[9]):
        need(any("s_cbranch_execz" in line for line in lines[max(0, position - 8):position]),
             "scale-load exec guard")
    iu4 = [line for line in body.splitlines() if "v_wmma_i32_16x16x32_iu4" in line]
    need(len(iu4) == 8, "native IU4 inventory")
    need(sum("neg_lo:[0,1,0]" in line for line in iu4) == 4, "unsigned A-low/signed W")
    need(sum("neg_lo:[1,1,0]" in line for line in iu4) == 4, "signed A-high/signed W")
    need(body.count("s_barrier_signal") == 2 and body.count("s_barrier_wait") == 2, "barrier inventory")
    need("global_load" not in "\n".join(line for line in body.splitlines()
                                        if "v_add_co_ci_u32" in line),
         "carry address")
    # Exact maxima over the only admitted shapes, including the final byte loaded.
    maxima = {name: [] for name in MAX_VOFFSETS}
    shapes = ((1024,5120),(4096,5120),(5120,6144),(5120,10240),
              (5120,17408),(7168,5120),(12288,5120),(34816,5120))
    for tokens in (1024,2048,4096,8192):
      for rows, columns in shapes:
        groups = columns // 64
        maxima["activation_code"].append(
            (tokens - 1) * (columns // 2) + (groups - 1) * 32 + 28)
        maxima["weight_code"].append(
            (((((rows - 1) >> 4) * groups + groups - 1) * 4 + 3) * 16 +
              ((rows - 1) & 15)) * 8)
        maxima["activation_scale"].append(
            ((tokens - 1) * groups + groups - 1) * 2)
        maxima["weight_scale"].append(
            ((((rows - 1) >> 4) * groups + groups - 1) * 16 +
              ((rows - 1) & 15)) * 2)
    actual_voffsets = {name: max(values) for name, values in maxima.items()}
    need(actual_voffsets == MAX_VOFFSETS, "u32 maxima changed")
    load_widths = {
        "activation_code": 4,
        "weight_code": 8,
        "activation_scale": 2,
        "weight_scale": 2,
    }
    actual_final_bytes = {
        name: offset + load_widths[name] - 1
        for name, offset in actual_voffsets.items()
    }
    need(actual_final_bytes == MAX_FINAL_BYTES, "final-byte maxima changed")
    need(all(value < 2**32 for value in actual_final_bytes.values()),
         "final loaded byte is not zero-extendable")
    if candidate_disassembly is not None or baseline_disassembly is not None or notes is not None:
        need(candidate_disassembly is not None and baseline_disassembly is not None and notes is not None,"complete loaded-object inputs")
        check_loaded_object(candidate_disassembly,baseline_disassembly,notes)
    result = {
        "logical_vgpr": vgpr,
        "architectural_vgpr": ((vgpr + 7) // 8) * 8,
        "lds_bytes": 17152,
        "compiler_occupancy_waves": 16,
        "iu4": 8,
        "qualified_tuples": 32,
    }
    for name in MAX_VOFFSETS:
        result[f"{name}_max_voffset"] = actual_voffsets[name]
        result[f"{name}_max_final_byte"] = actual_final_bytes[name]
    return result


def snapshot(path: Path, label: str) -> str:
    absolute = Path(os.path.abspath(path))
    metadata = os.lstat(absolute)
    need(stat.S_ISREG(metadata.st_mode) and not absolute.is_symlink(),
         f"{label} must be a regular non-symlink")
    with absolute.open("rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    need(identity(before) == identity(after) == identity(os.stat(absolute)),
         f"{label} changed while read")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--product-executable", type=Path, required=True)
    args = parser.parse_args()
    source_sha = snapshot(args.source, "product source")
    profile_sha = snapshot(args.profile, "dispatch profile")
    assembly_sha = snapshot(args.assembly, "compiler listing")
    executable_sha = snapshot(args.product_executable, "product qualifier")
    objdump="/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump";readobj="/opt/rocm/core-10.0/lib/llvm/bin/llvm-readobj"
    def invoke(command:list[str])->str:
        value=subprocess.run(command,text=True,capture_output=True,check=False)
        need(value.returncode==0,"object inspection failed")
        return value.stdout
    with tempfile.TemporaryDirectory(prefix="ninfer-scalar-base-static-") as directory:
        code_object = Path(directory) / "product.hsaco"
        capture = extract(args.product_executable, code_object, code_symbol=SYMBOL)
        candidate=invoke([objdump,"-d","--mcpu=gfx1201",str(code_object)])
        baseline=candidate
        notes=invoke([readobj,"--symbols","--notes",str(code_object)])
    need(len(re.findall(rf"(?m)^[0-9a-f]+ <{re.escape(SYMBOL)}>:$",candidate))==1,"unique candidate symbol")
    need(len(re.findall(rf"(?m)^\s+Name:\s+{re.escape(SYMBOL)}\s+\(",notes))==1,"unique candidate ELF symbol")
    result = check(args.source.read_text(), args.assembly.read_text(),candidate,baseline,notes)
    result.update({
        "source_sha256": source_sha,
        "profile_sha256": profile_sha,
        "assembly_sha256": assembly_sha,
        "product_executable_sha256": executable_sha,
        "loaded_code_object_sha256": capture["code_object_sha256"],
        "loaded_disassembly_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "loaded_notes_sha256": hashlib.sha256(notes.encode()).hexdigest(),
    })
    print("PASS " + " ".join(f"{key}={value}" for key,value in result.items()) +
          " scratch=0 spills=0 scalar_bases=5")


if __name__ == "__main__":
    main()
