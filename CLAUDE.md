# emberwave — CLAUDE.md

A portable (wgpu) port of the Pyrowave low-latency intra-frame wavelet video codec. 이 파일은 Claude Code가 이 저장소를 처음부터 끝까지 자율적으로 개발·검증·문서화·배포하기 위한 지침이다. 사람의 개입 없이 진행하되, 아래 "중단 조건"에 해당하면 멈추고 보고한다.

## 0. 한 줄 목표

Themaister의 Pyrowave(Vulkan 전용 GPU 웨이블릿 코덱, MIT)의 디코더를 Rust + wgpu + WGSL로 재구현해서 Vulkan / Metal / DX12 / WebGPU 어디서든 같은 코드로 돌아가게 만든다. 1차 완성 기준은 **"원본 Pyrowave 인코더가 만든 비트스트림을 파일에서 읽어 레퍼런스 디코더와 픽셀 단위로 일치하는 프레임을 뽑고, macOS(Metal)에서 4K 프레임당 디코딩 시간을 측정해 README에 표로 기록"**이다.

## 1. 배경 (왜 만드는가)

Pyrowave는 인코딩/디코딩을 GPU 컴퓨트로 처리해 프레임당 1ms 미만의 지연을 내는 LAN 게임 스트리밍용 코덱이다. Valve가 Steam Remote Play 베타에 채택했고, Sunshine/Moonlight 포크(Solarflare, Aurora, MoreOrLessSoftware)에 붙어 있다.

그러나 디코더가 Vulkan 컴퓨트 셰이더 전용이라 macOS / iOS / 브라우저에는 클라이언트가 없다. 이 프로젝트는 그 구멍을 메운다.

참고 저장소 (읽기용, 코드 복사 금지 — 로직을 이해해서 WGSL/Rust로 재작성한다):

- 원본: https://github.com/Themaister/pyrowave (MIT)
- WiVRn 포크 (인코더 통합 참고): https://github.com/WiVRn/WiVRn
- Aurora-qt (Moonlight 클라이언트 통합 방식 참고): https://github.com/Koloses/aurora-qt
- Solarflare (호스트 쪽 패킷화/FEC 참고): https://github.com/Koloses/Solarflare
- 설계 해설 글(개발자 블로그): "I designed my own ridiculously fast game streaming video codec – PyroWave" — 검색해서 읽을 것.

## 2. 범위

### 포함 (MUST)

- Pyrowave 비트스트림 파서 (Rust, GPU 무관)
- 역 DWT(이산 웨이블릿 변환) 컴퓨트 파이프라인 (WGSL)
- 역양자화 / 엔트로피 복호 단계 — 원본이 GPU에서 하는 단계는 GPU에서, CPU에서 하는 단계는 CPU에서 동일하게 분리
- YUV → RGB 변환 셰이더 (4:2:0 및 원본이 지원하는 서브샘플링 전부)
- CLI 도구 `emberwave-cli`: `.pyro`(또는 원본 포맷의 확장자) 파일 → PNG/Y4M 시퀀스 출력, 프레임당 디코딩 시간 출력
- 레퍼런스 일치 테스트: 원본 디코더 출력과 PSNR ∞(비트 일치) 또는 부동소수 오차로 설명 가능한 PSNR ≥ 60dB
- 벤치마크: 1080p / 1440p / 4K 각 60프레임 시퀀스, macOS Metal에서 평균·p99 디코딩 시간
- 문서: README(영문 기본, 한국어 요약 포함), ARCHITECTURE.md, BENCHMARKS.md, CONTRIBUTING.md
- CI: GitHub Actions — Linux(Vulkan/lavapipe 소프트웨어 렌더러로 테스트), macOS(Metal), Windows(DX12). 최소한 빌드 + 파서 유닛 테스트는 3개 OS 전부 통과.

### 포함 (SHOULD — 시간이 남으면)

- 인코더 포팅 (디코더 완성 후). 완성 시 round-trip 테스트 추가.
- WebGPU 데모: wasm 빌드 + 브라우저에서 파일 드롭 → 재생하는 정적 페이지 (GitHub Pages).
- Moonlight 연동용 C FFI (`emberwave-ffi` crate, cbindgen으로 헤더 생성).

### 제외 (MUST NOT)

- 네트워크 전송 / RTP / FEC — 코덱 범위 밖. 이슈로만 남긴다.
- 원본 저장소의 셰이더·C++ 코드를 복사·붙여넣기하지 않는다. 라이선스상 가능하지만 이 프로젝트의 가치는 "재구현"에 있다. 알고리즘 구조와 비트스트림 포맷만 동일하게 맞춘다.
- Vulkan 전용 확장 기능에 의존하지 않는다. wgpu가 공통으로 보장하는 기능만 사용.

## 3. 기술 스택 (고정)

| 항목 | 선택 | 비고 |
|---|---|---|
| 언어 | Rust (stable, 최신) | edition 2021 이상 |
| GPU | wgpu (최신 안정 버전) | 백엔드 자동 선택, 테스트에서 강제 지정 가능하게 |
| 셰이더 | WGSL | naga로 컴파일 검증을 CI에 포함 |
| 워크스페이스 | cargo workspace | `emberwave-core`(파서+GPU), `emberwave-cli`, `emberwave-ffi`(SHOULD), `emberwave-web`(SHOULD) |
| 테스트 | cargo test + 스냅샷/PSNR 테스트 | 픽셀 비교는 image crate |
| 벤치 | criterion 또는 자체 타이머 (GPU 타임스탬프 쿼리 사용 가능하면 그걸로) | 결과를 BENCHMARKS.md에 자동 갱신하는 스크립트 포함 |
| 린트 | clippy -D warnings, rustfmt | CI 게이트 |
| 라이선스 | MIT | 원본 Pyrowave MIT 고지 포함 |

## 4. 저장소 구조 (목표)

```
emberwave/
├─ CLAUDE.md                ← 이 파일
├─ README.md
├─ ARCHITECTURE.md
├─ BENCHMARKS.md
├─ CONTRIBUTING.md
├─ LICENSE                  ← MIT + Pyrowave 고지
├─ Cargo.toml               ← workspace
├─ crates/
│  ├─ emberwave-core/
│  │  ├─ src/
│  │  │  ├─ bitstream/      ← 파서 (순수 Rust, no_std 가능하면 좋음)
│  │  │  ├─ gpu/            ← wgpu 파이프라인, 버퍼 관리
│  │  │  ├─ shaders/        ← *.wgsl
│  │  │  └─ lib.rs
│  │  └─ tests/             ← 레퍼런스 일치 테스트
│  ├─ emberwave-cli/
│  ├─ emberwave-ffi/        (SHOULD)
│  └─ emberwave-web/        (SHOULD)
├─ testdata/
│  ├─ README.md             ← 테스트 벡터 생성 방법
│  └─ *.pyro + *.ref.png    ← 소형 벡터만 커밋(수 MB 이하), 대형은 스크립트로 생성
├─ scripts/
│  ├─ gen_testdata.sh       ← 원본 pyrowave 빌드 → 벡터 생성
│  └─ bench.sh
└─ .github/workflows/ci.yml
```

## 5. 마일스톤 & 작업 순서

각 마일스톤은 GitHub Issue 1개 + 브랜치 1개 + PR 1개로 진행한다. PR은 CI 통과 후 self-merge. 마일스톤 안에서는 작은 커밋을 자주 한다(Conventional Commits).

### M0 — 부트스트랩 (Issue #1)

- [ ] cargo workspace 생성, 빈 crate 3개(core/cli/ffi는 placeholder)
- [ ] CI 골격 (3 OS 빌드 + fmt + clippy)
- [ ] README 초안: 목표·현재 상태 배지·로드맵 체크리스트
- [ ] LICENSE
- [ ] testdata/README.md에 원본 Pyrowave 빌드 방법 기록

완료 기준: `cargo build --workspace` 3 OS 통과, PR 머지

### M1 — 비트스트림 포맷 역공학 & 파서 (Issue #2)

- [ ] 원본 소스에서 비트스트림 레이아웃(헤더, 서브밴드 구조, 양자화 파라미터, 패킷 경계) 파악 → ARCHITECTURE.md에 표와 그림으로 문서화
- [ ] Rust 파서 구현. 입력: 바이트 슬라이스. 출력: 프레임 메타 + 서브밴드 계수 버퍼(또는 GPU로 넘길 압축 블록)
- [ ] 원본 인코더로 만든 벡터 3개(작은 해상도, 단색 / 그라데이션 / 노이즈)로 파서 유닛 테스트
- [ ] 퍼징 테스트 1개 (cargo fuzz 또는 proptest) — 잘못된 입력에 패닉하지 않을 것

완료 기준: 테스트 통과, ARCHITECTURE.md에 포맷 설명 완료

### M2 — GPU 디코드 파이프라인 (Issue #3)

- [ ] wgpu 디바이스 초기화 추상화 (백엔드 강제 옵션, headless 가능)
- [ ] WGSL 셰이더: 역양자화 → 역 DWT(레벨별 멀티패스) → YUV 평면 출력
- [ ] 원본이 CPU에서 하는 전처리가 있으면 그대로 CPU에 둔다 (불필요한 "최적화" 금지 — 먼저 일치가 목표)
- [ ] 1프레임 디코딩 API: `Decoder::decode(&[u8]) -> Frame`
- [ ] 레퍼런스 일치 테스트: 벡터 3개 PSNR 측정, 기준(≥60dB 또는 비트 일치) 미달 시 원인 분석을 이슈 코멘트로 남길 것

완료 기준: 3개 벡터 모두 기준 통과

### M3 — 색변환 & CLI (Issue #4)

- [ ] YUV→RGB WGSL (BT.709 기본, 원본이 BT.601/Full-range 옵션이 있으면 플래그로)
- [ ] `emberwave-cli decode <in> --out-dir <dir> [--format png|y4m] [--backend vulkan|metal|dx12]`
- [ ] `emberwave-cli bench <in> [--frames N]` → 평균/p50/p99 ms 출력, JSON 옵션
- [ ] `emberwave-cli info <in>` → 헤더 덤프

완료 기준: CLI로 4K 벡터 디코딩해 PNG 저장 성공

### M4 — 벤치마크 & 문서 (Issue #5)

- [ ] 1080p / 1440p / 4K × 60프레임 벡터 생성 스크립트 (커밋은 하지 않고 scripts로 재생성)
- [ ] macOS Metal에서 측정 → BENCHMARKS.md (환경: 칩, OS, wgpu 버전, 날짜 명시)
- [ ] 가능하면 Linux lavapipe(소프트웨어)도 참고치로 기록
- [ ] README 완성: 설치 / 사용법 / 벤치 표 / 로드맵 / 한국어 요약 섹션
- [ ] CONTRIBUTING.md

완료 기준: README만 읽고 처음 보는 사람이 10분 안에 디코딩 성공 가능

### M5 — 안정화 (Issue #6)

- [ ] 4K 벡터로 메모리 누수·버퍼 재사용 확인 (반복 디코딩 1000프레임)
- [ ] 에러 타입 정리 (thiserror), panic 제거
- [ ] 릴리스 태그 v0.1.0 + GitHub Release (CLI 바이너리 3 OS 첨부)

완료 기준: Release 발행

### M6+ (SHOULD) — 인코더 / WebGPU 데모 / FFI

각각 Issue로 먼저 남기고, M5까지 끝난 뒤 시간이 남으면 진행. 진행 시 같은 규칙(Issue→브랜치→PR).

## 6. 작업 규칙 (자율 동작 프로토콜)

### Git / GitHub

- 기본 브랜치 main. 직접 커밋 금지. `feat/m1-bitstream-parser` 같은 브랜치에서 작업 → PR.
- 커밋 메시지: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `perf:`, `refactor:`). 본문에 왜를 쓴다.
- PR 본문 템플릿: 목적 / 변경 요약 / 테스트 방법 / 벤치 영향(있으면) / 관련 Issue(`Closes #N`).
- CI가 초록이어야 머지. 빨간 상태로 머지 금지.
- Issue는 마일스톤별 1개 + 발견한 문제마다 별도 생성. 라벨: milestone, bug, perf, docs, question.
- gh CLI 사용. 인증이 안 돼 있으면 중단 조건.

### 코드 품질

- `cargo fmt`, `cargo clippy --all-targets -- -D warnings` 통과 전 커밋 금지.
- 공개 API에는 doc comment. 셰이더 파일 상단에 해당 단계가 원본 알고리즘의 어느 부분인지 주석.
- 매직 넘버 금지 — 비트스트림 상수는 const로 이름 붙여서, 원본 소스의 어디서 왔는지 주석.

### 테스트

- 새 기능은 테스트 없이 머지하지 않는다.
- GPU 테스트는 CI에서 소프트웨어 백엔드로 돌리고, 하드웨어 수치는 로컬 벤치 스크립트로만 생성.
- 테스트 벡터는 원본 Pyrowave 인코더로 생성한다. 직접 만든 가짜 벡터로 "통과"시키지 않는다.

### 문서

- 결정 사항은 ARCHITECTURE.md의 "Decisions" 섹션에 날짜와 함께 기록 (ADR 축약형).
- 벤치 수치는 환경 없이 적지 않는다.
- README의 로드맵 체크박스는 마일스톤 머지 때마다 갱신.

### 하지 말 것

- 원본 코드 복사.
- 일치 테스트가 안 맞는데 임계값을 낮춰서 통과시키기.
- "나중에 고침" 주석 남기고 넘어가기 → 대신 Issue 생성.
- 의존성 무분별 추가. 새 crate 추가 시 PR 본문에 이유.

## 7. 중단 조건 (멈추고 사용자에게 보고)

다음 중 하나면 작업을 멈추고 현재 상태·원인·선택지를 정리해 보고한다:

1. `gh auth status` 실패 등 GitHub 권한 문제
2. 원본 Pyrowave 빌드가 로컬에서 안 돼서 테스트 벡터를 만들 수 없음 (환경 문제 해결 시도 2회까지)
3. 비트스트림 포맷에 문서화되지 않은 부분이 있어 역공학으로 확정이 안 됨 → 추정 가설과 검증 방법을 적고 보고
4. 레퍼런스 일치 테스트가 2회 이상의 근본 원인 분석 후에도 기준 미달
5. wgpu가 보장하지 않는 기능이 필수로 보이는 경우(예: 서브그룹 연산) → 대안 설계 제시 후 보고
6. 라이선스·저작권 판단이 필요한 경우

그 외 판단은 전부 자율적으로 내리고, 내린 판단은 PR 본문이나 ARCHITECTURE.md Decisions에 남긴다.

## 8. 환경 정보

- 개발 머신: macOS (Apple Silicon, M3) — Metal 백엔드. Vulkan 테스트는 MoltenVK 또는 CI Linux.
- 참고용 호스트 PC: Windows, RTX 4070 Super (원본 Pyrowave 인코더로 벡터 생성 가능, 필요 시 사용자에게 요청)
- 테스트 벡터 생성이 macOS에서 막히면(원본이 Vulkan 전용) MoltenVK로 시도하고, 안 되면 중단 조건 2.

## 9. 완료 정의 (Definition of Done)

- [ ] M0~M5 Issue 전부 Closed, PR 머지
- [ ] CI 3 OS 초록
- [ ] `emberwave-cli decode`로 4K 벡터 → PNG 출력, 레퍼런스와 PSNR 기준 통과
- [ ] BENCHMARKS.md에 macOS Metal 1080p/1440p/4K 수치
- [ ] README 영문 + 한국어 요약, 설치부터 실행까지 재현 가능
- [ ] v0.1.0 Release 발행
- [ ] 남은 SHOULD 항목은 Issue로 등록되어 있음

위 체크리스트가 전부 채워지면 "완료"를 선언하고 최종 보고서(무엇을 했고, 수치가 얼마고, 남은 Issue가 뭔지)를 출력한다.

---

> 진행 중 확인된 사실과 그에 따른 결정(상류 Metal 포트의 존재, 비트스트림 문서 동결, PWV1 파일 포맷 등)은 이 파일을 고치지 않고 `ARCHITECTURE.md`의 Decisions 섹션에 기록한다.
