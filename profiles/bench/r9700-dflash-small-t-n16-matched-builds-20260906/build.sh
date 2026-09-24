#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-small-t-n16-matched-builds-20260906"
control="$repo/build-r9700-dflash-small-t-n16-cd966d72-control-20260906"
candidate="$repo/build-r9700-dflash-small-t-n16-cd966d72-candidate-20260906"
source_checkout=/ssdpool2nvme/local_llm/ninfer-amd-r9700-cd966d72-dflash-build-source

test "$PWD" = "$repo"
test "$#" -eq 0
sha256sum -c "$package/prepared.sha256"
test ! -e "$control"
test ! -e "$candidate"
test ! -e "$source_checkout"
test "$(git rev-parse 'cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce^{tree}')" = a8cf2ffcef1646d2c39128eab72c5808c889cbcd
test "$(sha256sum /usr/bin/cmake | cut -d' ' -f1)" = 1c5227af4edd22d8d689def545e18ee458260c0fd579eba2187967f38817e638
test "$(sha256sum /usr/bin/make | cut -d' ' -f1)" = d78b8f1d099fbcfb6f2f49ab87223b9b68fb3956642f92d6ec6de812e8afa965
test "$(sha256sum /opt/rocm/llvm/bin/clang++ | cut -d' ' -f1)" = 241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247
test "$(sha256sum /usr/bin/c++ | cut -d' ' -f1)" = 1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769

git worktree add --detach "$source_checkout" cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce
test "$(git -C "$source_checkout" rev-parse HEAD)" = cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce
test "$(git -C "$source_checkout" rev-parse 'HEAD^{tree}')" = a8cf2ffcef1646d2c39128eab72c5808c889cbcd
test -z "$(git -C "$source_checkout" status --porcelain)"

configure() {
  local build=$1
  local selector=$2
  /usr/bin/cmake -S "$source_checkout" -B "$build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DNINFER_BUILD_APPS=ON \
    -DNINFER_BUILD_BENCHMARKS=ON \
    -DNINFER_BUILD_R9700_CORE_QUALIFIER=ON \
    -DNINFER_R9700_KV_VALUE_GROUP=16 \
    -DNINFER_R9700_Q4_ACTIVATION_BITS=8 \
    -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
    -DNINFER_R9700_FP8_QK_WMMA=1 \
    -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF \
    -DNINFER_R9700_XATTENTION_STRIDE=16 \
    -DNINFER_R9700_XATTENTION_TAU_PERMILLE=1000 \
    -DNINFER_R9700_DFLASH_SMALL_T_CANDIDATE="$selector"
}

configure "$control" 0
configure "$candidate" 1

for build in "$control" "$candidate"; do
  /usr/bin/cmake --build "$build" --parallel 4 --target \
    ninfer_bench ninfer_bench_support_test ninfer_r9700_linear_prefill_dispatch_test
  HIP_VISIBLE_DEVICES=-1 /usr/bin/ctest --test-dir "$build" --output-on-failure \
    -R '^(ninfer_bench_support_test|ninfer_r9700_linear_prefill_dispatch_test)$'
done
