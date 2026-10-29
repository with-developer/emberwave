<p align="center"><a href="README.md">English</a> | <b>한국어</b></p>

<p align="center">
  <img src="docs/images/banner.svg" alt="emberwave: wgpu로 다시 쓴 PyroWave 웨이블릿 비디오 코덱" width="100%">
</p>

<p align="center">
  <a href="https://github.com/with-developer/emberwave/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/with-developer/emberwave/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/with-developer/emberwave/releases/latest"><img alt="최신 릴리스" src="https://img.shields.io/github/v/release/with-developer/emberwave?color=e8302a"></a>
  <a href="LICENSE"><img alt="라이선스: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://with-developer.github.io/emberwave/"><img alt="브라우저에서 써 보기" src="https://img.shields.io/badge/try_it-WebGPU_player-ff7a1a"></a>
</p>

<p align="center">
  <b><a href="https://with-developer.github.io/emberwave/">브라우저에서 스트림 재생</a></b> ·
  <a href="https://with-developer.github.io/emberwave/bench.html">내 기기로 벤치마크</a> ·
  <a href="BENCHMARKS.md">벤치마크</a> ·
  <a href="crates/emberwave-ffi/README.md">C API</a> ·
  <a href="crates/emberwave-ffi/examples/ios">iPhone 앱</a>
</p>

[PyroWave](https://github.com/Themaister/pyrowave)는 로컬 네트워크에서 게임을 스트리밍하기 위한 비디오 코덱입니다. 대역폭을 더 쓰는 대신, GPU가 프레임 하나를 1~2 ms 안에 인코딩하고 디코딩합니다. Valve가 Steam 클라이언트 베타의 Steam Remote Play에 넣었습니다([GamingOnLinux](https://www.gamingonlinux.com/2026/10/steam-client-beta-for-october-1st-brings-many-remote-play-pyrowave-improvements/)). 업스트림 구현은 Vulkan 컴퓨트 셰이더로 쓰였고, Metal 포트가 따로 있습니다. **emberwave는 PyroWave를 [wgpu](https://wgpu.rs) 위에서 Rust와 WGSL로 다시 쓴 것입니다.** 하나의 코드와 같은 비트스트림으로 Metal, Vulkan, DX12, WebGPU에서 돌아가고, 네이티브 앱부터 브라우저 탭과 iPhone까지 갑니다.

- **빠릅니다.** Apple M3 Max에서 프레임을 연달아 처리하면, 4K 프레임 하나를 평면까지 0.90 ms, RGB까지 1.22 ms에 디코딩하고, GPU에서 CPU 메모리부터 CPU 메모리까지 1.90 ms에 인코딩합니다. 업스트림의 Metal 포트는 같은 FP32 정밀도로 0.80 ms, 기본값인 FP16으로 0.56 ms에 디코딩하고, 인코딩에는 3.2~3.3 ms가 걸립니다([`BENCHMARKS.md`](BENCHMARKS.md), 2026-10-26).
- **프레임마다 독립적입니다.** 다른 프레임을 참조하는 프레임이 없어서, 패킷을 잃어도 그 프레임 하나만 잃고 키프레임을 기다리지 않습니다. 프레임마다 정해진 바이트 예산을 지킵니다.
- **업스트림과 양방향으로 호환됩니다.** 업스트림 인코더가 만든 스트림을 업스트림 디코더와 같게, PSNR 64 dB 이상으로 디코딩하고, 업스트림 디코더는 emberwave의 스트림을 읽습니다. 프레임당 바이트가 같으면, 2026년 10월에 레이트 컨트롤을 고친 업스트림 인코더와 emberwave 인코더는 비슷합니다. emberwave의 luma PSNR은 업스트림보다 그라데이션에서 0.8 dB 낮은 것부터 테스트 장면에서 2.3 dB 높은 것까지이고, 노이즈에서는 같습니다.
- **네이티브로, 브라우저에서, iPhone에서.** 명령줄 도구, 페이지 안에서 WebGPU로 디코딩하고, 스스로 인코딩한 데모 스트림으로 시작하며, 네트워크가 전하듯 순서가 섞이고 일부가 사라진 패킷으로도 스트림을 재생하는 [브라우저 플레이어](https://with-developer.github.io/emberwave/), Linux와 Windows(x86_64), macOS(Apple silicon)용으로 미리 빌드한 라이브러리가 있는 C API, iPhone·iPad·Mac 앱용 XCFramework가 있습니다. Apple GPU에서 테스트했고, CI에서는 소프트웨어 Vulkan과 DX12에서 테스트합니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/decode-dark.svg">
  <img alt="GPU 메모리의 RGB까지 프레임당 디코딩 시간. Apple M3 Max: Metal 네이티브로 1080p 0.53 ms, 4K 1.22 ms, Chrome의 WebGPU로 0.70~1.00 ms와 1.69~1.88 ms. iPhone 17: 네이티브 앱에서 1.62~1.63 ms와 6.21~6.29 ms, Safari에서 2.36 ms와 8.05 ms." src="docs/images/decode-light.svg" width="100%">
</picture>

이 시간은 프레임을 연달아 처리할 때의 시간입니다. 스트림의 프레임은 초당 60개씩 도착하고, 그 사이에 GPU가 쉬면서 느려지므로 디코딩마다 더 오래 걸립니다([성능](#성능)).

> **상태: 0.x.** 비트스트림은 업스트림의 것이고, Rust와 C API는 마이너 버전 사이에 아직 바뀔 수 있습니다. CI는 모든 변경을 Linux에서 lavapipe Vulkan 드라이버로, macOS에서 Metal로, Windows에서 WARP 위의 DX12로 테스트합니다. 릴리스와 그 변경 사항은 [`CHANGELOG.md`](CHANGELOG.md)에 있습니다.

## 화질

<p align="center">
  <img alt="게임 붉은사막의 4K 프레임과, 그중 돌벽 앞 소나무가 있는 320×180 영역을 두 배로 키운 것. 원본, 그리고 네 가지 예산 안에서 만든 emberwave 프레임을 디코딩한 것이며, 초당 60프레임 기준 비트레이트와 프레임 전체의 luma PSNR은 322 Mbit/s 25.8 dB, 499 Mbit/s 27.1 dB, 816 Mbit/s 29.9 dB, 1200 Mbit/s 32.9 dB입니다." src="docs/images/quality.png" width="100%">
</p>

PyroWave는 대역폭을 들여 속도를 얻고, 대역폭이 얼마나 드는지는 그림에 따라 다릅니다. 이 게임의 4K 프레임은 대부분이 풀, 솔잎, 돌이라 어려운 그림입니다. 프레임은 emberwave 인코더가 만들었습니다. 디코딩한 패널마다 초당 60프레임 기준 비트레이트와 프레임 전체의 luma PSNR을 적었고, 가장 많이 잃는 영역을 두 배로 키워 보여 줍니다. 322 Mbit/s에서는 솔잎이 뭉개지고 벽의 이음매가 흐려집니다. 816 Mbit/s에서는 대부분 돌아오고, 1.2 Gbit/s에서는 그 영역이 원본에 가깝습니다. 벤치마크의 합성 4K 장면은 첫 그림이 322 Mbit/s에서도 42.7 dB를 유지합니다. 이 프레임에서 Apple 하드웨어 HEVC 인코더의 키프레임은 실시간 모드 밖에서 같은 예산 안에 luma PSNR이 1.9~3.5 dB 높고(Apple M3 Max, macOS 27.0.1, 2026-10-27), HEVC 스트림은 시작할 때와 손실 뒤에만 키프레임이 필요합니다. PyroWave는 대역폭이 넉넉한 로컬 네트워크를 위한 코덱입니다([`BENCHMARKS.md`](BENCHMARKS.md#picture-quality)).

<sub>프레임은 붉은사막(Crimson Desert)의 것이며 © Pearl Abyss입니다. 본 콘텐츠는 펄어비스의 공식 자료가 아니며, 본 콘텐츠에는 펄어비스가 권리를 보유하고 있는 상표 또는 저작물이 포함되어 있습니다. 펄어비스가 만들거나 보증한 것이 아닙니다. 이 그림에는 이 저장소의 라이선스가 적용되지 않습니다.</sub>

## 작동 방식

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/pipeline-dark.svg">
  <img alt="Apple M3 Max에서 emberwave를 거치는 4K 프레임 하나. GPU 인코딩: 웨이블릿 변환 0.40 ms, 양자화 0.59 ms, 레이트 제어 0.13 ms, 패킹 0.18 ms, 최대 1.2 MB의 프레임. 패킷이 네트워크를 건넙니다. 디코딩: CPU에서 조립 0.10 ms, 역양자화 0.19 ms, 역변환 0.40 ms, YUV에서 RGB로 0.24 ms." src="docs/images/pipeline-light.svg" width="100%">
</picture>

프레임은 그림을 웨이블릿 변환한 것입니다. 그 계수를 블록 단위로 양자화해서 비트 평면으로, 가장 중요한 비트부터 보냅니다. 프레임을 예산에 맞추려고 인코더는 그림을 가장 덜 해치는 곳에서 가장 덜 중요한 비트 평면을 버리며, 대역마다 눈이 얼마나 민감한지로 가중합니다. 그림의 단계 중 디코더가 패킷을 조립하는 단계만 빼고 모두 WGSL 컴퓨트 셰이더입니다. 조립은 업스트림 디코더처럼 CPU에서 합니다. 그 앞뒤로 CPU가 평면을 들여보내고, 프레임을 꺼내고, 그림의 헤더를 복사합니다. 비트스트림과 패스들, 그 뒤의 결정은 [`ARCHITECTURE.md`](ARCHITECTURE.md)에 있습니다.

## 빠른 시작

[Rust](https://rustup.rs) 1.87 이상과 Vulkan, Metal, DX12 중 하나를 지원하는 GPU가 필요합니다. GPU가 없으면 소프트웨어 드라이버로도 됩니다. Linux에서는 Mesa의 lavapipe(Debian과 Ubuntu에서는 `sudo apt install mesa-vulkan-drivers libvulkan1`), Windows에서는 내장된 WARP입니다.

```sh
git clone https://github.com/with-developer/emberwave.git
cd emberwave
cargo build --release -p emberwave-cli
target/release/emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out
```

처음 빌드는 몇 분 걸립니다. 마지막 명령은 사용하는 어댑터와 프레임마다의 시간을 출력하고, `out/gradient_200x120_420_000000.png`부터 `000002.png`까지를 남깁니다. 색 그라데이션입니다. `testdata/`의 `.pwv1` 파일은 업스트림 인코더가 만든 작은 스트림입니다.

Rust 없이 쓰려면 [릴리스](https://github.com/with-developer/emberwave/releases)에서 Linux(x86_64), macOS(Apple Silicon), Windows(x86_64)용 `emberwave-cli`를 받아 `testdata/`의 `.pwv1` 파일을 디코딩하면 됩니다. macOS 바이너리는 서명되지 않았으므로, 받은 뒤 `xattr -d com.apple.quarantine emberwave-cli`를 한 번 실행하세요.

## 사용법

`emberwave-cli`는 PWV1 형식(`.pwv1`)의 PyroWave 파일을 디코딩합니다. 아래 예는 `emberwave-cli`가 `PATH`에 있다고 가정합니다(`cargo install --path crates/emberwave-cli`가 그곳에 설치합니다. 아니면 `target/release/emberwave-cli`를 쓰세요).

```sh
# 파일에 무엇이 들어 있는지
emberwave-cli info testdata/gradient_200x120_420.pwv1

# 프레임마다 PNG 하나로, 또는 YUV 평면의 Y4M 비디오로 디코딩
emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out
emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out --format y4m

# 디코더의 시간 재기. --output rgb는 색 변환을 포함하고, --json은 JSON으로 출력
emberwave-cli bench testdata/gradient_200x120_420.pwv1 --output rgb

# 두 Y4M 파일을 평면별로 비교. PSNR이 60 dB 미만인 평면이 있으면 실패
emberwave-cli compare out/gradient_200x120_420.y4m testdata/gradient_200x120_420.ref.y4m --min-psnr 60

# Y4M 비디오(8비트 4:2:0 또는 4:4:4)를 프레임마다 6000바이트 안으로 인코딩.
# --gpu는 GPU에서 인코딩
emberwave-cli encode testdata/encoder/scene_200x120_420.source.y4m --out scene.pwv1 --bytes-per-frame 6000 --gpu

# 인코더의 시간 재기, 여기서는 GPU에서
emberwave-cli bench-encode testdata/encoder/scene_200x120_420.source.y4m --bytes-per-frame 6000 --gpu
```

디코더와 `--gpu`를 준 인코더는 기계에 있는 GPU를 씁니다. Vulkan, Metal, DX12, 또는 Mesa의 lavapipe나 Windows의 WARP 같은 소프트웨어 드라이버입니다. `--backend vulkan|metal|dx12`와 `--software`, 그리고 환경 변수 `EMBERWAVE_BACKEND`로 선택을 좁힐 수 있고, 사용하는 어댑터를 출력합니다. RGB 변환은 파일이 알려 주는 대로(행렬, 범위, 크로마 위치) 하며, `--matrix bt601|bt709|bt2020`과 `--range full|limited`로 바꿀 수 있습니다. 모든 옵션과 시간을 재는 방법은 [`ARCHITECTURE.md`](ARCHITECTURE.md#command-line-tool)에 있습니다.

### C에서 쓰기

C API는 클라이언트가 받는 대로 패킷을 호출자의 메모리로 디코딩하고, 호스트가 보내는 대로 그림을 프레임과 패킷으로 인코딩합니다. 헤더 `crates/emberwave-ffi/include/emberwave.h`와 공유 또는 정적 라이브러리로 되어 있고, 릴리스마다 미리 빌드한 라이브러리도 올립니다. 빌드하고 링크하는 방법은 [`crates/emberwave-ffi/README.md`](crates/emberwave-ffi/README.md)에 있고, [`examples/c/decode.c`](crates/emberwave-ffi/examples/c/decode.c)와 [`examples/c/encode.c`](crates/emberwave-ffi/examples/c/encode.c)는 완전한 프로그램입니다.

iPhone, iPad, Mac 앱을 위해 릴리스마다 정적 라이브러리를 XCFramework인 `Emberwave.xcframework`로도 올리며, Swift에서는 `Emberwave`로 import합니다. Metal 클라이언트는 그림을 GPU의 텍스처 하나에 그리게 하거나 Y, Cb, Cr 평면을 텍스처 세 장에 복사하게 하고, 그 `id<MTLTexture>`를 읽을 수 있습니다([GPU의 그림](crates/emberwave-ffi/README.md#pictures-on-the-gpu-metal)). [`examples/swift`](crates/emberwave-ffi/examples/swift)는 Swift에서 이것으로 파일을 디코딩하고, [`examples/ios`](crates/emberwave-ffi/examples/ios)는 iPhone과 iPad용 벤치마크 앱이며, CI는 예제와 앱을 이것으로 iOS 시뮬레이터에서 돌립니다([iPhone, iPad, Mac 앱](crates/emberwave-ffi/README.md#iphone-ipad-and-mac-apps)).

## 성능

<!-- bench-summary:begin (scripts/bench.sh가 생성합니다. 고쳐도 덮어씁니다) -->
Apple M3 Max(Metal)에서 잰 프레임당 디코딩 시간입니다. RGB 출력은 GPU 메모리에 두었고, 2026-10-26에 측정했으며, 단위는 밀리초입니다.

| 스트림 | 평균 | p99 |
|---|---|---|
| 1080p | 0.53 | 0.79 |
| 1440p | 0.69 | 0.97 |
| 4K | 1.22 | 1.62 |

같은 기계에서는 업스트림의 Metal 디코더가 더 빠릅니다. 4K 한 프레임에 FP32로 0.80 ms가 걸리고, emberwave는 0.90 ms가 걸립니다(평면까지만). 측정 방법과 전체 수치는 [`BENCHMARKS.md`](BENCHMARKS.md)에 있습니다.

4K 한 프레임을 1.2 MB 안에 인코딩하는 데 GPU에서 1.90 ms가 걸립니다. CPU 메모리의 평면에서 CPU 메모리의 프레임까지의 시간이며, 업스트림 인코더는 FP32로 3.33 ms가 걸립니다.
<!-- bench-summary:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/encode-dark.svg">
  <img alt="Apple M3 Max에서 4K 프레임 하나를 인코딩하는 시간: GPU의 emberwave 1.90 ms, 업스트림 PyroWave FP16 3.17 ms와 FP32 3.33 ms, 80 Mbit/s의 Apple HEVC 인코더 12.76~13.00 ms." src="docs/images/encode-light.svg" width="100%">
</picture>

iPhone 17의 Safari에서 [벤치마크 페이지](https://with-developer.github.io/emberwave/bench.html)는 1080p 프레임을 2.4 ms, 4K 프레임을 8.1 ms에 GPU 메모리의 RGB로 디코딩합니다([`BENCHMARKS.md`](BENCHMARKS.md#in-the-browser-phones-included)). XCFramework를 쓰는 네이티브 [벤치마크 앱](crates/emberwave-ffi/examples/ios)은 같은 프레임을 Metal 텍스처로 1.6 ms와 6.2~6.3 ms, Y·Cb·Cr 텍스처로 1.3 ms와 4.8 ms에 디코딩합니다([`BENCHMARKS.md`](BENCHMARKS.md#on-the-iphone-natively)).

같은 앱이 Moonlight가 쓰는 하드웨어 HEVC 디코더도 Moonlight의 기본 비트레이트로 잽니다. 프레임을 연달아 디코딩하면 emberwave가 텍스처로 디코딩하는 속도가 M3 Max에서 2.2~3.9배(4K에서 1.0 ms 대 2.6~3.4 ms), iPhone 17에서 1.4~1.7배(4.5 ms 대 HEVC의 6.2~6.3 ms) 빠릅니다. 위 측정의 하루 뒤에 잰 값입니다. 프레임이 초당 60개씩 오면 GPU는 프레임 사이에 쉬면서 느려지지만 미디어 엔진은 그렇지 않습니다. 그러면 Mac에서는 둘이 비슷하게 걸리고, 폰에서는 HEVC가 1.5~2.7배 빠릅니다([`BENCHMARKS.md`](BENCHMARKS.md#against-hevc-on-the-same-device)).

## 목표

- 원본 PyroWave 인코더가 만든 비트스트림을 파일에서 바로 디코딩하고, 레퍼런스 디코더의 출력과 일치시킵니다. PyroWave 디코딩은 부동소수점 연산이라 설계상 비트 단위로 같지 않으므로, 합격 기준은 레퍼런스 디코더에 대해 평면마다 PSNR 60 dB 이상입니다.
- wgpu가 모든 백엔드에서 보장하는 기능만 씁니다. Vulkan 전용 확장도, 서브그룹 연산도 쓰지 않습니다.
- macOS(Metal)에서 1080p / 1440p / 4K의 프레임당 디코딩 시간을 재고, 그 수치를 측정 환경과 함께 공개합니다.

범위 밖: 네트워크 전송, RTP, FEC.

## 로드맵

끝난 일: 첫 릴리스까지의 마일스톤 M0~M5([#1](https://github.com/with-developer/emberwave/issues/1)~[#6](https://github.com/with-developer/emberwave/issues/6)), 그 뒤로 CPU와 GPU의 인코더([#14](https://github.com/with-developer/emberwave/issues/14)), 브라우저 플레이어와 벤치마크 페이지([#15](https://github.com/with-developer/emberwave/issues/15)), 클라이언트와 호스트를 위한 C API([#16](https://github.com/with-developer/emberwave/issues/16), [#42](https://github.com/with-developer/emberwave/issues/42)), Metal 텍스처를 포함해 GPU 메모리의 그림을 주고받기([#27](https://github.com/with-developer/emberwave/issues/27), [#33](https://github.com/with-developer/emberwave/issues/33)), iPhone·iPad·Mac 앱용 XCFramework([#52](https://github.com/with-developer/emberwave/issues/52)). 0.12에서는 Apple GPU에서 프레임마다 하던 복사 두 가지를 없앴습니다. Metal에서는 평면을 텍스처에 바로 쓰고, Apple silicon에서는 프레임을 셰이더가 읽는 곳에 바로 씁니다([#86](https://github.com/with-developer/emberwave/issues/86)). 릴리스마다의 변경은 [`CHANGELOG.md`](CHANGELOG.md)에 있습니다.

다음은 이 순서로 합니다.

1. **0.13, 실제 사용에서 재기.** 클라이언트의 화면 출력 루프 안에서 디코딩해 화면까지의 시간을 Mac과 iPhone에서 잽니다([#81](https://github.com/with-developer/emberwave/issues/81)). NVENC의 HEVC와 AV1 스트림을 emberwave의 스트림 옆에서 디코딩하고, 움직이는 게임 영상의 화질을 같은 비트레이트에서 예측 프레임을 쓰는 HEVC와 비교합니다([#82](https://github.com/with-developer/emberwave/issues/82)). XCFramework를 Swift 패키지로도 냅니다. [#86](https://github.com/with-developer/emberwave/issues/86)의 마지막 항목인 프레임마다의 CPU 준비도 줄입니다.
2. **0.14, 화질과 플랫폼.** 같은 비트스트림 안에서 더 나은 인코더를 만들고, Linux와 Windows의 클라이언트와 호스트를 위해 C API의 GPU 연동에 Vulkan과 Direct3D 12 핸들을 넣습니다.
3. **1.0을 향해, 클라이언트.** emberwave가 코덱으로 남도록 별도 저장소에서 브라우저로 스트리밍합니다. 호스트의 화면을 GPU에서 인코딩해 로컬 네트워크로 보내고, 탭에서 WebGPU로 디코딩하며, 먼저 지연 시간부터 잽니다. 그다음 PyroWave를 스트리밍하는 호스트들에 emberwave 인코더를 제안합니다. 1.0에는 안정된 C와 Rust API, 실제로 쓰이는 클라이언트 하나, 그리고 스트림의 속도에서 업스트림만큼 빠른 디코딩이 필요합니다.

## 저장소 구성

README 말고는 모든 문서가 영어로 되어 있습니다.

| 경로 | 내용 |
|---|---|
| `crates/emberwave-core` | 비트스트림 파서, CPU 레퍼런스 디코더, 색 변환, wgpu/WGSL 디코더, CPU와 GPU의 인코더 |
| `crates/emberwave-cli` | 명령줄 도구 `emberwave-cli` |
| `crates/emberwave-ffi` | C API: 헤더, 라이브러리, C 예제 |
| `crates/emberwave-web`, `web/` | 브라우저 플레이어와 그 벤치마크: WebAssembly 모듈과 페이지 |
| `testdata/` | 테스트 벡터와 그 재생성 방법 |
| `scripts/` | 테스트 데이터, 4K 합격 검사, 벤치마크 스크립트 |
| `docs/images/` | README의 배너와 차트(`scripts/readme_figures.py`가 `BENCHMARKS.md`의 수치로 그림), 화질 그림(`scripts/quality_figure.py`가 만듦) |
| `README.md` | 이 README의 영어판 |
| `ARCHITECTURE.md` | 비트스트림 형식, GPU 파이프라인, 결정 기록 |
| `BENCHMARKS.md` | 디코딩 시간을 재는 방법과 그 결과 |
| `CONTRIBUTING.md` | 코드를 빌드하고 테스트하고 바꾸는 방법 |
| `CHANGELOG.md` | 릴리스마다 바뀐 것 |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | MIT 라이선스, PyroWave의 라이선스와 이 라이선스가 적용되지 않는 것 |

## 개발

```sh
cargo build --workspace
cargo test --workspace
```

어댑터가 없으면 GPU 테스트는 건너뜁니다. `EMBERWAVE_REQUIRE_GPU`가 설정되어 있으면 건너뛰지 않습니다. 테스트 벡터, 규칙, 변경이 들어가는 과정은 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

## 업스트림과의 관계

emberwave는 PyroWave의 알고리즘 구조와 [비트스트림 정의](https://github.com/Themaister/pyrowave/blob/7fe0e2667cd3076b8b2bf04bcb80bc8d691b800b/bitstream/bitstream.md)를 따르지만, 업스트림 저장소에서 복사한 코드는 없습니다. 테스트 벡터와 레퍼런스 출력은 업스트림 자신의 인코더와 디코더가 만듭니다. [`testdata/README.md`](testdata/README.md)를 보세요.

업스트림에는 Apple 전용 Metal 포트도 있습니다. emberwave는 Vulkan, DX12, 브라우저까지 아우르는 하나의 이식 가능한 코드를 목표로 합니다.

## 라이선스

MIT입니다. [`LICENSE`](LICENSE)를 보세요. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에는 PyroWave의 저작권 고지와 라이선스, 그리고 이 라이선스가 적용되지 않는 것이 있습니다. `docs/images/quality.png`의 붉은사막 프레임은 © Pearl Abyss이며, 릴리스의 소스 아카이브에는 들어가지 않습니다.
