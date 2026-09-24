#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-rmsnorm-rows56-matched-builds-20260906"
source_dir=/ssdpool2nvme/local_llm/ninfer-amd-r9700-dflash-rmsnorm-rows56-build-source
control=/ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700-dflash-rmsnorm-rows56-control-20260906
candidate=/ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700-dflash-rmsnorm-rows56-candidate-20260906

test "$PWD" = "$repo"
test "$#" -eq 0
sha256sum -c "$package/prepared.sha256"
test -z "$(git status --porcelain)"
test ! -e "$source_dir" && test ! -L "$source_dir"
test ! -e "$control" && test ! -L "$control"
test ! -e "$candidate" && test ! -L "$candidate"
test ! -e "$package/build-receipt.json" && test ! -L "$package/build-receipt.json"
commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')
git worktree add --detach "$source_dir" "$commit"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test "$(git -C "$source_dir" rev-parse 'HEAD^{tree}')" = "$tree"
test -z "$(git -C "$source_dir" status --porcelain)"
make -C "$source_dir/tools/r9700" rmsnorm-dflash-rows56-production-build -j2

common=(
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON
  -DNINFER_BUILD_R9700_CORE_QUALIFIER=ON -DNINFER_R9700_KV_VALUE_GROUP=16
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF
  -DNINFER_R9700_XATTENTION_STRIDE=16 -DNINFER_R9700_XATTENTION_TAU_PERMILLE=1000
  -DNINFER_R9700_DFLASH_SMALL_T_CANDIDATE=0
  -DNINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=1
)
cmake -S "$source_dir" -B "$control" "${common[@]}" \
  -DNINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=0
cmake -S "$source_dir" -B "$candidate" "${common[@]}" \
  -DNINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=1
cmake --build "$control" --target ninfer_bench ninfer_bench_support_test -j2
cmake --build "$candidate" --target ninfer_bench ninfer_bench_support_test -j2
HIP_VISIBLE_DEVICES=-1 "$control/tests/ninfer_bench_support_test"
HIP_VISIBLE_DEVICES=-1 "$candidate/tests/ninfer_bench_support_test"
/usr/bin/python3 "$package/publish.py" --commit "$commit" --tree "$tree"
/usr/bin/python3 "$package/validate.py"
