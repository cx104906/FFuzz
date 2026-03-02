# FFuzz

## Compile PUT with pass in conda env (e.g. ptc)
```
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CC="/usr/bin/clang-18 -fpass-plugin=xxx/cxpass1.so -Wl,--as-needed,xxx/cxfuncs1.so -fuse-ld=lld -g -Wno-unused-command-line-argument" CXX="/usr/bin/clang++-18 -fpass-plugin=xxx/cxpass1.so -Wl,--as-needed,xxx/cxfuncs1.so -fuse-ld=lld -g -Wno-unused-command-line-argument" MAX_JOBS=5 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=OFF BUILD_TEST=0 USE_DISTRIBUTED=0 python -m pip install --no-build-isolation -v -e .
```

## Also compile PUT for asan or csan in conda env (pta/ptn)
```
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CC="/usr/bin/clang-18 -fsanitize=address -fuse-ld=lld -g -Wno-unused-command-line-argument" CXX="/usr/bin/clang++-18 -fsanitize=address -fuse-ld=lld -g -Wno-unused-command-line-argument" MAX_JOBS=5 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=OFF BUILD_TEST=0 USE_DISTRIBUTED=0 python -m pip install --no-build-isolation -v -e .
:
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CC="/usr/bin/clang-12 -fuse-ld=lld -g -Wno-unused-command-line-argument" CXX="/usr/bin/clang++-12 -fuse-ld=lld -g -Wno-unused-command-line-argument" MAX_JOBS=5 USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=OFF BUILD_TEST=0 USE_DISTRIBUTED=0 python -m pip install --no-build-isolation -v -e .
```

## Provide target API to test
```
python cxfuzz.py --o xxx/output-test --a torch.nn.Conv3d
```
