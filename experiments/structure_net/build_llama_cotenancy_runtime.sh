#!/usr/bin/env bash
# Reproduce the pinned standalone GGUF runtime without modifying the Torch env.
set -euo pipefail
study_root="data/experiments/llama_tensorized_cotenancy"
vendor_root="$study_root/llama.cpp"
runtime_revision="23b0202a189c44a54625aadcb37a946dd1d6278d"
mkdir -p "$study_root"
if [[ ! -d "$study_root/tools_env" ]]; then
  python3 -m venv "$study_root/tools_env"
  "$study_root/tools_env/bin/python" -m pip install cmake==3.31.6 ninja==1.11.1.4
fi
if [[ ! -d "$vendor_root" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$vendor_root"
  git -C "$vendor_root" checkout --detach "$runtime_revision"
fi
if [[ "$(git -C "$vendor_root" rev-parse HEAD)" != "$runtime_revision" ]]; then
  echo "Unexpected llama.cpp revision; preserve it and choose a different path." >&2
  exit 1
fi
"$study_root/tools_env/bin/cmake" -S "$vendor_root" -B "$vendor_root/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/$study_root/tools_env/bin/ninja" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF \
  -DCMAKE_BUILD_TYPE=Release
"$study_root/tools_env/bin/cmake" --build "$vendor_root/build" --target llama llama-bench llama-simple -j4
g++ -O3 -std=c++17 -pthread \
  -I "$vendor_root/include" -I "$vendor_root/ggml/include" -I "$vendor_root/vendor" \
  -I /usr/local/cuda/include experiments/structure_net/llama_cotenancy_gguf.cpp \
  -L "$vendor_root/build/bin" -L /usr/local/cuda/lib64 \
  -Wl,-rpath,"$PWD/$vendor_root/build/bin" -Wl,-rpath,/usr/local/cuda/lib64 \
  -lllama -lggml -lggml-base -lcudart -o "$study_root/gguf_assay"
