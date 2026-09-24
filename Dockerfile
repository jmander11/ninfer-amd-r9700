# syntax=docker/dockerfile:1
# Explicit maintainer contexts: --build-context rocm=/opt/rocm/core-10.0
# --build-context python311=/path/to/self-contained-python-3.11
FROM ubuntu:24.04 AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates libstdc++6 libnuma1 libdrm2 libelf1 libzstd1 libtinfo6 \
    libexpat1 libffi8 libssl3t64 zlib1g libbz2-1.0 liblzma5 libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rocm / /opt/rocm/core-10.0/
ENV ROCM_PATH=/opt/rocm/core-10.0
ENV HIP_PATH=/opt/rocm/core-10.0
ENV PATH=/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/lib/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/llvm/lib:/opt/rocm/core-10.0/lib/rocm_sysdeps/lib
LABEL org.ninfer.platform=gfx1201

FROM toolchain AS build
ARG NINFER_BUILD_JOBS=4
RUN apt-get update && apt-get install --yes --no-install-recommends \
    build-essential cmake ninja-build git patch pkg-config util-linux \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libcurl4-openssl-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=python311 / /opt/python311/
ENV PATH=/opt/python311/bin:$PATH
ENV NINFER_PYTHON=/opt/python311/bin/python3.11
RUN /opt/python311/bin/python3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11)'
WORKDIR /src
COPY . .
RUN case "$NINFER_BUILD_JOBS" in 1|2|3|4|5|6|7|8|9|10|11|12|13|14) ;; *) echo 'NINFER_BUILD_JOBS must be 1..14' >&2; exit 2;; esac \
    && cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel "$NINFER_BUILD_JOBS" --target ninfer ninfer-serve ninfer-ppl

FROM toolchain AS runtime
RUN apt-get update && apt-get install --yes --no-install-recommends \
    curl libavcodec60 libavformat60 libavutil58 libswscale7 libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=build /build/apps/ninfer-ppl /usr/local/bin/ninfer-ppl
WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM
CMD ["ninfer-serve", "--help"]
