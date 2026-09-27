# Mac Studio M5 Ultra (256GB / 64-core GPU / 2TB SSD) 로컬 LLM 가이드

작성일: 2026-09-27

## 0. 하드웨어 근거 (왜 이 모델들이 되는가)

| 항목 | 값 | LLM 관점 의미 |
|---|---|---|
| 통합 메모리 | 256GB | 모델 가중치 + KV 캐시 + OS가 전부 여기 들어감. 실사용 상한은 **~200~220GB** (아래 참고) |
| 메모리 대역폭 | **1.2TB/s** (M3 Ultra 819GB/s 대비 +50%) | 디코드(토큰 생성) 속도 = 대역폭 ÷ "토큰당 읽는 바이트". MoE는 활성 파라미터만 읽으므로 빠름 |
| GPU | 64코어 + 코어당 Neural Accelerator | 프리필(프롬프트 처리)이 연산 바운드 → M3 Ultra 대비 프리필 대폭 개선. 디코드는 대역폭 바운드라 64 vs 80코어 차이 작음 |
| SSD | 2TB | 모델 파일 100~180GB짜리 여러 개 + oMLX SSD KV 캐시. 4~5개 대형 모델이 한계 |

**메모리 예산 계산법**
- macOS 기본 GPU wired 한도 ≈ 전체의 75% (≈192GB). `sudo sysctl iogpu.wired_limit_mb=230000` 정도로 올리면 ~230GB까지 GPU에 할당 가능.
- 안전 공식: `모델 파일 크기 + KV 캐시(컨텍스트 길이에 비례) + 10~15GB(OS/앱) ≤ 220GB`
- 따라서 **모델 파일 ≤ 180GB**면 편안, **180~215GB**면 짧은 컨텍스트만, **220GB 초과**면 불가.

**속도 추정 근거**: 디코드 tok/s ≈ 1.2TB/s ÷ (활성 파라미터 바이트 + KV 읽기). 4bit 기준 활성 6B ≈ 3GB → 이론상 수백 tok/s, 실측은 오버헤드로 1/3~1/2. M3 Ultra 실측치 × 1.3~1.5배가 M5 Ultra 기대치.

---

## 1. 추천 LLM 리스트 (256GB 기준)

### Tier 1 — 이 머신의 주력 (대형 MoE, 근-프론티어)

| 모델 | 구조 | 권장 양자화 / 크기 | 예상 속도 | 용도 / 근거 |
|---|---|---|---|---|
| **GLM-5.3-Flash** | 320B-A18B MoE (Kimi-Delta 선형어텐션 34층 + DSA 11층, 288 expert) | MLX **4bit ≈ 177.5GB** (mixed 4_8 ≈ 190GB, 6bit/8bit는 256GB에 안 들어감) | M3 Ultra 실측 18~25 tok/s → M5 Ultra 25~35 tok/s 기대 | 256GB 티어의 **코딩 대표 모델**. 8bit→6bit 차이는 노이즈 수준, 4bit는 약간의 perplexity 손실. 이미 쓰던 4_6 mixed는 ~200GB라 컨텍스트 여유가 적음 → **4bit 또는 4_8 mixed 권장** |
| **Qwen3.8-Flash-Next** | 125B-A6B 하이브리드 Gated-DeltaNet/sparse-attn MoE + 51B n-gram 임베딩 (총 ~180B) | 4bit 105GB / mixed 4_8 ~107GB / **6bit 158GB** / 8bit ~200GB(빡빡) | **M5 Ultra 실측 85~100 tok/s**, 64K~256K 컨텍스트에서도 60~85 | **속도·품질 밸런스 1위**. KV 캐시 토큰당 24KiB → 1M 컨텍스트가 24.6GB. 6bit 유지해도 100GB 여유 → 다른 모델과 동시 로드 가능 |
| **DeepSeek V4-Flash (0731) / V4.1-Flash** | 284B-A13B MoE | MLX **4bit ≈ 152GB** (mixed ≈ 93GB, 6bit ≈ 220GB는 부적합) | Mac Studio 커뮤니티 25~35 tok/s | 범용 추론·코딩. mlx-lm의 `deepseek_v4` 지원 버전 필요. 1M 컨텍스트 |
| **MiniMax M2.5 / M2.7** | 230B-A10B MoE | MLX 4bit ≈ 125~130GB | 30~45 tok/s 기대 | SWE-bench Verified 80.2% — **에이전틱 코딩** 특화. `mlx-community/MiniMax-M2.7-4bit`, `-4bit-mxfp4` 존재 |
| **gpt-oss-120b** | 117B-A5B MoE | MXFP4 ≈ 65GB | 40~45 tok/s (512GB M5 Ultra 실측 43) | 가볍고 빠른 추론 모델. 다른 대형 모델과 **동시 상주** 가능 |
| **Qwen3.5-122B-A10B** | 122B-A10B MoE | 4bit ≈ 65GB / 8bit ≈ 125GB | ~35 tok/s | 품질 점수 89/100. 8bit로 무손실급 운용 가능 |

**경계선 (들어가긴 함)**
- **Qwen3.5-397B-A17B** — MXFP4/4bit ≈ 210~220GB. wired limit 올리면 로드되지만 KV 여유 거의 없음 → 짧은 컨텍스트 전용. 실사용은 비추.

### Tier 2 — Dense / 멀티모달 / 보조 (작지만 파라미터당 품질 높음)

| 모델 | 크기 | 근거 |
|---|---|---|
| **Qwen3.8-27B** (또는 3.6-27B) | 8bit ≈ 29GB | Dense 하이브리드 어텐션, 비전+thinking+tool, SWE-bench 77.2%. Dense라 MoE보다 느림(30~40 tok/s) — **MTPLX/oMLX의 MTP(멀티토큰 예측)** 쓰면 1.6배. 이미 쓰던 모델 — 그대로 유지 가치 있음 |
| **Gemma 4 31B** | 8bit ≈ 33GB, Q4 ≈ 17.5GB | Apache 2.0, LMArena 1452 (Qwen3.5-397B보다 높음), 비전 지원. 범용 글쓰기/추론 |
| **Maple-Preview 20B-A1B** (DeepGrove) | ternary ≈ 5.3GB | Apple Silicon 218~575 tok/s. 초고속 보조/드래프트/분류용 |

### 안 들어가는 것 (참고)
- Kimi K2.5 / K2.7-Code (1T-A32B): 4bit 600GB+ → 2×512GB 필요
- GLM-5.3 (Flash 아님, 4bit 418GB), DeepSeek V4 풀 모델, Qwen3.5-397B 8bit

### 추천 조합 (256GB 안에서 동시 상주)
1. **코딩 에이전트 세트**: GLM-5.3-Flash 4bit (177GB) 단독 + KV 여유 40GB
2. **속도 세트**: Qwen3.8-Flash-Next 6bit (158GB) + gpt-oss-120b (65GB) = 223GB → wired limit 상향 필요, 아니면 Flash-Next 4bit(105GB)+gpt-oss
3. **범용 세트**: DeepSeek V4-Flash 4bit (152GB) + Qwen3.8-27B 8bit (29GB) + Maple (5GB)

---

## 2. 서버/런타임 비교 (어디서 돌릴 수 있나)

| 런타임 | 엔진 | API | 강점 | 약점 | 대형 신형 모델(GLM-5.3-Flash, Qwen3.8-Flash-Next) 지원 |
|---|---|---|---|---|---|
| **mlx-lm / mlx-vlm** (CLI) | MLX 네이티브 | `mlx_lm.server` OpenAI 호환 | **신규 아키텍처를 가장 먼저 지원** (glm5_next, qwen4_exp, deepseek_v4). 레퍼런스 구현 | GUI 없음, 단일 요청 위주 | ◎ 최신 pip/git 버전 필요 |
| **oMLX** (v0.6.4, 2026-08-29) | mlx-lm 기반 | **OpenAI + Anthropic(`/v1/messages`)** | **SSD KV 캐시**(재시작 후에도 프리픽스 복원 → 에이전트 TTFT 대폭 단축), continuous batching, MTP, 메뉴바 앱, 모델 LRU/핀 관리, Claude Code 최적화 | MLX 포맷만, GGUF 불가 | ◎ (자체 MTP quant도 배포) |
| **LM Studio 0.4.x** | MLX 엔진 + llama.cpp | OpenAI 호환 | GUI, 모델 브라우저, 레이어별 오프로드, 가장 쉬움. 독립 벤치에서 MLX 엔진이 Ollama보다 ~46% 빠름 | **신형 아키텍처 지원이 늦음** (Qwen3.8-Flash-Next 이슈 #2407 미해결) | △ 엔진 업데이트 대기 필요 |
| **Ollama ≥0.19** | Apple Silicon에서 MLX 백엔드(32GB+) | OpenAI 호환 | Docker/스크립트 친화, 라이브러리 관리 편함 | **자체 라이브러리 변환본만** 사용 → 신형 모델 lag. `qwen3.8:27b-mlx`는 v0.32.12부터 | △ Flash-Next/GLM-5.3-Flash 미확인 |
| **Rapid-MLX** | MLX 네이티브 | OpenAI + Anthropic | continuous batching, radix prefix cache, **int4/int8 KV 캐시**, 27종 tool-call 파서. 동시 8스트림에서 Ollama 대비 3× | 단일 스트림은 1.5×, dense는 이득 없음 | ○ 자체 quant 배포 (`rapid-mlx/Qwen3.8-Flash-Next-4bit`) |
| **MTPLX** | MLX + 네이티브 MTP | OpenAI + Anthropic | Qwen3.8 계열 **MTP 투기 디코딩** (M5 Max에서 27B 125 tok/s), 온도 무관 exact | Qwen3.8/Bonsai 계열 전용 | ○ Flash-Next, 27B 전용 팩 |
| **pMLX** | MLX, 런타임 동적 양자화 | - | 130~150 tok/s 주장, NVMe n-gram 스트리밍 | **2026-09-01 기준 미출시** | 대기 |
| **llama.cpp / Unsloth Desktop** (GGUF) | Metal | OpenAI 호환 | MLX 변환본 없을 때 폴백, Unsloth UD 동적 양자화 품질 좋음 | Apple Silicon에서 MLX보다 느림 | ○ GGUF는 항상 있음 |

### 결론적 추천
- **주 서버: oMLX** — Claude Code/OpenCode 같은 에이전트 워크플로우에서 SSD KV 캐시가 결정적. Anthropic API 호환이라 `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`만 바꾸면 됨.
- **보조: LM Studio** — 모델 탐색/비교/채팅 GUI. 신형 모델은 엔진 업데이트 후.
- **실험: mlx-lm git HEAD** — 새 아키텍처 나오면 가장 먼저 여기서 돌아감.
- Ollama는 이 머신 용도(대형 MoE 최신 모델)에는 라이브러리 lag 때문에 비추.

---

## 3. MLX 모델 정리 (모델별 권장 리포 + 작동 서버)

| 모델 | 추천 MLX 리포 | 크기 | oMLX | LM Studio | mlx-lm | Rapid-MLX | MTPLX | Ollama |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash 4bit | `pipenetwork/GLM-5.3-Flash-MLX-4bit` | 177.5GB | ◎ (MTP quant 있음) | △ | ◎ (glm5_next) | ○ | ✕ | ✕ |
| GLM-5.3-Flash mixed 4_8 | `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` | ~190GB | ◎ | △ | ◎ | ○ | ✕ | ✕ |
| Qwen3.8-Flash-Next 6bit | `pipenetwork/Qwen3.8-Flash-Next-MLX-6bit` | 158GB | ◎ | △ (이슈 open) | ◎ (qwen4_exp) | ○ | ◎ (자체 팩) | ✕ |
| Qwen3.8-Flash-Next 4bit | `pipenetwork/…-MLX-4bit`, `rapid-mlx/Qwen3.8-Flash-Next-4bit`, `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` | 105GB | ◎ | △ | ◎ | ◎ | ◎ | ✕ |
| DeepSeek V4-Flash 4bit | `mlx-community/DeepSeek-V4-Flash-4bit` | 152GB | ◎ | △ | ◎ (deepseek_v4) | ○ | ✕ | ✕ |
| DeepSeek V4.1-Flash mixed 4_8 | `pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit` | ~93GB | ◎ | △ | ◎ | ○ | ✕ | ✕ |
| MiniMax M2.7 4bit | `mlx-community/MiniMax-M2.7-4bit` / `-4bit-mxfp4` | ~130GB | ◎ | ◎ | ◎ | ○ | ✕ | ○ |
| gpt-oss-120b | `lmstudio-community/gpt-oss-120b-MLX-8bit` / mxfp4 | ~65GB | ◎ | ◎ | ◎ | ◎ | ✕ | ◎ |
| Qwen3.5-122B-A10B | `mlx-community/Qwen3.5-122B-A10B-MLX-4bit` (8bit도 있음) | 65 / 125GB | ◎ | ◎ | ◎ | ◎ | ✕ | ○ |
| Qwen3.8-27B 8bit | `lmstudio-community/Qwen3.6-27B-MLX-8bit`, `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality` | ~29GB | ◎ (MTP) | ◎ | ◎ | ◎ | ◎ | ◎ (`qwen3.8:27b-mlx`) |
| Gemma 4 31B | `mlx-community/gemma-4-31b-it-8bit` | ~33GB | ◎ | ◎ | ◎ (mlx-vlm) | ◎ | ✕ | ◎ |
| Maple-Preview | `deepgrove/maple-preview-2bit-mlx`, `txgsync/Maple-Preview-oQ8e` | ~5GB | ◎ | ◎ | ◎ | ○ | ✕ | ○ |

◎ 확인됨 / ○ 가능(일반 MLX 경로) / △ 엔진 버전 의존·이슈 있음 / ✕ 미지원·미확인

### MLX 운용 팁
1. **mlx-lm 버전 확인** — GLM-5.3-Flash(`glm5_next`), Qwen3.8-Flash-Next(`qwen4_exp`), DeepSeek V4(`deepseek_v4`)는 2026년 8~9월 버전 필요. `pip install -U mlx-lm mlx-vlm` 또는 git HEAD.
2. **양자화 선택 원칙** — 8bit≈6bit(노이즈 수준) > mixed 4_8(expert만 4bit) > 4bit(약간 손실). 256GB에선 **"4bit로 최대 모델" vs "6~8bit로 한 단계 작은 모델"** 중 후자가 대개 품질이 낫고 빠름 (Flash-Next 6bit가 GLM 4bit보다 3배 빠름).
3. **KV 캐시** — 긴 컨텍스트는 Rapid-MLX의 int8 KV 캐시나 oMLX SSD 캐시로 메모리 절약.
4. **wired limit** — `/etc/sysctl.conf` 또는 launchd로 `iogpu.wired_limit_mb=225000` 정도 영구 설정.
5. **MTP 활용** — Qwen3.8 계열과 GLM-5.3-Flash는 MTP 헤드 보유 → oMLX/MTPLX에서 1.5~2× 디코드 가속.

---

## 출처
- Apple 뉴스룸 M5 Max/Ultra 발표 (1.2TB/s): https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
- LLMCheck M5 Ultra 256GB 랭킹: https://llmcheck.net/best-llm/mac-studio-m5-ultra-256gb/
- ContextStudios M5 Ultra 가이드: https://www.contextstudios.ai/blog/mac-studio-m5-ultra-local-ai-guide
- OpenClawDC 256GB 추천: https://openclawdc.com/blog/best-local-llm-m5-ultra-256gb/
- OpenClawDC 런타임 비교: https://openclawdc.com/blog/ollama-vs-lm-studio-vs-llama-cpp-vs-omlx/
- QDNA Qwen3.8-Flash-Next on M5 Ultra: https://qdna.fr/en/blog/qwen3-8-flash-next-mlx-mac-studio-m5-ultra
- QDNA GLM-5.3-Flash on M5 Ultra: https://qdna.fr/en/blog/glm-5-3-flash-mlx-mac-studio-m5-ultra
- pipenetwork GLM-5.3-Flash MLX: https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit
- pipenetwork Qwen3.8-Flash-Next MLX 6bit: https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-6bit
- mlx-community DeepSeek-V4-Flash-4bit (152GB): https://llm-explorer.com/model/mlx-community%2FDeepSeek-V4-Flash-4bit,2PonA1JyUKnZGuqgVkk4RE
- mlx-community MiniMax-M2.7-4bit: https://huggingface.co/mlx-community/MiniMax-M2.7-4bit
- Qwen3.5-397B 로컬 가이드: https://unsloth.ai/docs/models/qwen3.5
- Unsloth Qwen3.8-Flash-Next: https://unsloth.ai/docs/models/qwen3.8-next
- Qwen3.6-27B MLX 8bit: https://huggingface.co/lmstudio-community/Qwen3.6-27B-MLX-8bit
- Gemma 4 on Apple Silicon: https://sudoall.com/gemma-4-31b-apple-silicon-local-guide/
- Maple-Preview: https://huggingface.co/deepgrove/maple-preview
- oMLX GitHub: https://github.com/jundot/omlx
- oMLX SSD KV 캐시 리뷰: https://andrew.ooo/posts/omlx-apple-silicon-ssd-kv-cache-review/
- mlx-lm vs oMLX 벤치: https://medium.com/macoclock/mlx-lm-vs-omlx-i-was-wrong-about-the-winner-8f36be328069
- Rapid-MLX: https://github.com/raullenchai/Rapid-MLX
- MTPLX: https://github.com/youssofal/MTPLX
- pMLX 발표 이슈: https://github.com/evanwtf/local-llm/issues/72
- Ollama MLX 백엔드: https://ollama.com/blog/mlx
- LM Studio Qwen3.8-Flash-Next 이슈: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2407
- Ollama vs LM Studio MLX 벤치: https://www.morphllm.com/comparisons/ollama-vs-lm-studio
- Kimi K2.5 / 512GB 적합성: https://pinggy.io/blog/self_hosting_llms_on_512gb_m5_ultra_mac_studio/
