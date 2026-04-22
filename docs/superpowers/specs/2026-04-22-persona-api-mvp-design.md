# Social Labs — Persona Agent API MVP 설계

**작성일**: 2026-04-22  
**상태**: 승인됨  
**목표**: B2B 리서치/컨설팅 시장을 위한 페르소나 에이전트 API + 데모 UI MVP

---

## 1. 제품 포지셔닝

### 핵심 가치 제안

> 패널 서베이 데이터를 **쿼리 가능한 살아있는 에이전트 군**으로 변환하는 인프라

- **NVIDIA Nemotron-Personas-Korea**: 합성 정적 스냅샷 → LLM 훈련용 데이터셋
- **Social Labs**: 22년 종단 궤적을 가진 에이전트 → 시뮬레이션/리서치 엔진

### 타겟 고객

**B2B 마케팅 리서치사 / HR 컨설팅 펌**  
- 기존 툴에 Social Labs API를 연결해 유사 페르소나 시뮬레이션을 자신들의 서비스에 통합
- 의사결정권자(파트너/디렉터)는 데모 UI로 설득, 개발팀은 API로 통합

### 현재 데이터 자산

- NLSY97 패널 (1997–2019, 100개 자서전 MD 파일)
- KuzuDB 지식 그래프 (563 노드 / 1,089 엣지)
- 확장 예정: HRS 2022, KEEP 한국 교육고용패널

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────┐
│              Social Labs MVP                    │
│                                                 │
│  ┌──────────────┐      ┌───────────────────┐    │
│  │  Demo UI     │      │  External Client  │    │
│  │  (Next.js)   │      │  (B2B 컨설팅사)   │    │
│  └──────┬───────┘      └────────┬──────────┘    │
│         └──────────┬────────────┘               │
│                    ▼                             │
│         ┌──────────────────────┐                │
│         │   FastAPI REST API   │                │
│         │   /v1/query          │                │
│         │   /v1/simulate       │                │
│         │   /v1/personas/{id}  │                │
│         └──────┬───────────────┘                │
│                │                                │
│    ┌───────────┼───────────┐                    │
│    ▼           ▼           ▼                    │
│  Embedding  KuzuDB      Claude API              │
│  시맨틱 검색  구조 필터    에이전트 응답           │
└─────────────────────────────────────────────────┘
```

**원칙**: Demo UI와 외부 B2B 클라이언트가 동일한 API를 호출. UI가 쇼케이스이자 API 사용 예시.

---

## 3. API 설계

### 인증
- `X-API-Key` 헤더
- MVP: 환경변수로 단순 관리 (`API_KEYS` 콤마 구분 리스트)

### 엔드포인트

#### `POST /v1/query` — 페르소나 검색

```json
// Request
{
  "context": "38세, 대졸, 마케팅 15년, 창업 고민 중",
  "filters": {
    "age_min": 35,
    "age_max": 45,
    "education": "Bachelor's Degree"
  },
  "n": 5
}

// Response
{
  "personas": [
    {
      "id": "ID_042",
      "match_score": 0.91,
      "summary": "40세 여성, 대졸, 중간소득 → 커리어 전환 경험",
      "key_events": ["2008 이직", "2012 소득 급락", "2015 회복"]
    }
  ]
}
```

#### `POST /v1/simulate` — 에이전트 응답 생성

```json
// Request
{
  "persona_ids": ["ID_042", "ID_067"],
  "question": "창업 전에 가장 후회한 결정은?",
  "mode": "individual"  // "individual" | "discussion"
}

// Response
{
  "responses": [
    { "id": "ID_042", "response": "저는 재정 준비 없이..." },
    { "id": "ID_067", "response": "네트워크를 과소평가했어요..." }
  ]
}
```

`mode: "discussion"` 은 연구자 모드로, 에이전트들이 서로의 응답을 참조하며 다중 라운드 대화. 기본 3라운드, 요청 시 `rounds` 파라미터로 조정 (최대 10).

#### `GET /v1/personas/{id}/trajectory` — 궤적 데이터

```json
{
  "id": "ID_042",
  "background": { "sex": "Female", "birth_year": 1981, "education": "Bachelor's" },
  "timeline": [
    { "year": 2000, "education": "Bachelor's", "income": 45000, "events": [] },
    { "year": 2008, "income": 12000, "events": ["divorce"] }
  ]
}
```

---

## 4. 페르소나 검색 엔진

### 2단계 검색

**1단계: 임베딩 시맨틱 검색**
- 자서전 100개를 사전 임베딩 → `output/embeddings.npy` 저장
- 사용자 컨텍스트 임베딩 → 코사인 유사도 → Top-20 후보
- 도구: `sentence-transformers` (로컬, 무료) — `all-MiniLM-L6-v2` 모델

**2단계: KuzuDB 구조 필터**
- Top-20에서 명시적 필터(`age_min`, `education` 등) 적용
- 기존 KuzuDB Cypher 쿼리 재활용
- 최종 N명 반환

| | 임베딩만 | KuzuDB만 | 2단계 조합 |
|--|--|--|--|
| 추상적 맥락 ("창업 고민") | ✅ | ❌ | ✅ |
| 정확한 조건 ("42세, 대졸") | ❌ | ✅ | ✅ |

### 임베딩 사전 처리
- 실행 시점: 서버 시작 시 1회 (`startup` 이벤트)
- 캐시: `output/embeddings.npy` 존재 시 로드, 없으면 생성

---

## 5. Demo UI

### 기술 스택
- 기존 Next.js 14 + TypeScript + Tailwind CSS 유지
- 신규 추가: `recharts` (궤적 차트)
- 모드 전환: 우측 상단 토글 `일반 | 연구자`

### 모드 A — 일반 사용자

```
┌─────────────────────────────────────────────────┐
│  [상황 입력창]                                   │
│  "저는 38세 마케터입니다. 창업을 고민 중인데..." │
│  [분석하기 →]                                   │
├────────────────┬────────────────────────────────┤
│  유사 페르소나  │  궤적 비교 차트 (recharts)      │
│                │  소득 / 학력 / 이벤트 타임라인  │
│  ● ID_042 91%  │                                │
│  ● ID_067 87%  │                                │
├────────────────┴────────────────────────────────┤
│  에이전트 조언                                   │
│  ID_042: "저도 비슷한 나이에..."                 │
└─────────────────────────────────────────────────┘
```

### 모드 B — 연구자

```
┌─────────────────────────────────────────────────┐
│  [시나리오 설정]  [페르소나 선택]  [시작]         │
├─────────────────────────────────────────────────┤
│  Round 1                                        │
│  ID_042: "이 상황에서 저라면..."                 │
│  ID_067: "ID_042의 말에 동의하지만..."           │
├─────────────────────────────────────────────────┤
│  [다음 라운드 →]    [JSON/CSV 내보내기]          │
└─────────────────────────────────────────────────┘
```

---

## 6. 기존 코드 변경 범위

| 파일 | 변경 내용 |
|------|----------|
| `backend/main.py` | OASIS WebSocket → FastAPI REST API로 재구성 |
| `backend/app.py` | `/v1/` 라우터 통합 또는 분리 유지 |
| `backend/` (신규) | `embeddings.py` — 임베딩 생성/로드/검색 |
| `backend/` (신규) | `persona_retriever.py` — 2단계 검색 로직 |
| `backend/` (신규) | `agent_runner.py` — 자서전 로드 + Claude API 호출 |
| `frontend/app/page.tsx` | 두 모드 UI로 재구성 |
| `frontend/` (신규) | `TrajectoryChart.tsx` — recharts 궤적 컴포넌트 |

---

## 7. 구현 우선순위

### Phase 2-A: API 백엔드 (1주)
1. `embeddings.py` — 자서전 임베딩 사전 처리
2. `persona_retriever.py` — 2단계 검색
3. `/v1/query` 엔드포인트
4. `agent_runner.py` — 자서전 → Claude 응답
5. `/v1/simulate` 엔드포인트 (individual 모드)
6. `/v1/personas/{id}/trajectory` 엔드포인트

### Phase 2-B: Demo UI (1주)
1. 모드 A — 입력창 + 페르소나 카드
2. `TrajectoryChart.tsx` — 궤적 시각화
3. 에이전트 조언 패널
4. 모드 B — 연구자 시뮬레이션 (discussion 모드)
5. JSON/CSV 내보내기

---

## 8. 미결 사항

| 항목 | 결정 필요 |
|------|----------|
| 임베딩 모델 | `sentence-transformers/all-MiniLM-L6-v2` vs Claude Embeddings API |
| API Key 관리 | 환경변수 단순 관리 (MVP) → 추후 DB 기반으로 전환 |
| 배포 환경 | 로컬 데모 vs Vercel/Railway 배포 |
| 과금 모델 | Per-query vs 월정액 구독 (MVP 이후 결정) |
