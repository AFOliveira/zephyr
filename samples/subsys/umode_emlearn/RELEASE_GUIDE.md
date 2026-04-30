# ET-SoC1 U-mode emlearn release guide

Run this on the ET-SoC1 board host.  Pick a local directory for the
release contents, then download and unpack the release bundle there.

```sh
export ROOT_DIR=/tmp/zephyr-etsoc1-emlearn
export RELEASE_TAG=etsoc1-umode-emlearn-2026-04-30
export RELEASE_URL=https://github.com/aifoundry-org/zephyr/releases/download/${RELEASE_TAG}

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

curl -L -O "${RELEASE_URL}/zephyr-etsoc1-emlearn-release.tar.gz"
tar -xzf zephyr-etsoc1-emlearn-release.tar.gz
```

Run the Zephyr ET-SoC1 U-mode emlearn demo on shire 0:

```sh
LD_LIBRARY_PATH="${ROOT_DIR}" "${ROOT_DIR}/basic_launcher" \
  --kernel_path="${ROOT_DIR}/zephyr_etsoc1_umode_emlearn.elf" \
  --device_type=silicon \
  --shire_mask=0x1 \
  --kernel_launch_timeout=30 \
  --num_launches=1
```

Expected output includes:

```text
loadKernel() kernel 0 loaded at 0x8006335000
[predictions] launch 0:
  iris (DecisionTree):
    [0] -> class 0 (setosa)
    [1] -> class 1 (versicolor)
    [2] -> class 2 (virginica)
```

The Zephyr source for this release is the same `RELEASE_TAG` in
`git@github.com:aifoundry-org/zephyr.git`.

## Build from source

Use this when you want to rebuild the Zephyr ELFs and release bundle
instead of using the prebuilt tarball.

```sh
export ROOT_DIR=${ROOT_DIR:-$HOME/zephyr-etsoc1-emlearn}
export RELEASE_TAG=etsoc1-umode-emlearn-2026-04-30
export ET_PLATFORM_COMMIT=261d33188cfc767607781fc476f93fc1d514ee2b
export ET_PLATFORM_REPO=ssh://git@github.com/vidas/et-platform.git

mkdir -p "${ROOT_DIR}/src" "${ROOT_DIR}/bundle"
cd "${ROOT_DIR}/src"

git clone git@github.com:aifoundry-org/zephyr.git zephyr
git -C zephyr fetch --depth=1 origin "refs/tags/${RELEASE_TAG}:refs/tags/${RELEASE_TAG}"
git -C zephyr checkout --detach "${RELEASE_TAG}"

git clone "${ET_PLATFORM_REPO}" et-platform
git -C et-platform fetch --depth=1 origin "${ET_PLATFORM_COMMIT}"
git -C et-platform checkout --detach "${ET_PLATFORM_COMMIT}"
```

Build the Zephyr ELFs:

```sh
cd "${ROOT_DIR}/src/zephyr"

west init -l .
west update
west zephyr-export

export ET_PLATFORM_ROOT="${ROOT_DIR}/src/et-platform"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export ZEPHYR_SDK_INSTALL_DIR=/home/afonso/toolchains/zephyr-sdk-0.17.4

west build \
  --build-dir build-etsoc1-umode-emlearn \
  -b etsoc1_minion_umode \
  samples/subsys/umode_emlearn \
  -p always \
  -- -DET_PLATFORM_ROOT="${ET_PLATFORM_ROOT}"

west build \
  --build-dir build-erbium-mmode-emlearn \
  -b erbium_minion \
  samples/subsys/umode_emlearn \
  -p always
```

Build the stock ET Platform launcher:

```sh
export ET_SDK_HOME=/usr/local/et
source "${ET_SDK_HOME}/.builds/host/conanrunenv-debug-x86_64.sh"

cd "${ROOT_DIR}/src/et-platform/gp-sdk/host"
mkdir -p build
cd build

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="${ET_SDK_HOME}/.builds/host/conan_toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . --target basic_launcher -j"$(nproc)"
```

Package the release bundle:

```sh
export LAUNCHER_BIN="${ROOT_DIR}/src/et-platform/gp-sdk/host/build/sdk/basic_launcher"

cp "${ROOT_DIR}/src/zephyr/build-etsoc1-umode-emlearn/zephyr/zephyr.elf" \
  "${ROOT_DIR}/bundle/zephyr_etsoc1_umode_emlearn.elf"
cp "${ROOT_DIR}/src/zephyr/build-erbium-mmode-emlearn/zephyr/zephyr.elf" \
  "${ROOT_DIR}/bundle/zephyr_erbium_mmode_emlearn.elf"
cp "${LAUNCHER_BIN}" "${ROOT_DIR}/bundle/"

ldd "${LAUNCHER_BIN}" \
  | awk '/=>/ { print $3 }' \
  | grep -E '/(libetrt|libdeviceLayer|libsw-sysemu|lib.*boost|libglog|libgflags|libunwind|liblz4|libgtest|libgmock|libdebugging|libthreadPool|libactionList|liblogging|libg3logger|libeasy_profiler|libbacktrace|libcap|libbz2|libz|liblzma)' \
  | xargs -r cp -t "${ROOT_DIR}/bundle/"

tar -czf "${ROOT_DIR}/zephyr-etsoc1-emlearn-release.tar.gz" \
  -C "${ROOT_DIR}/bundle" .
```
