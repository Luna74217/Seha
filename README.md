# Soha AI 챗봇

감정 기반 AI 에이전트 Soha와 대화할 수 있는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **감정 기반 응답**: Soha의 감정 상태에 따른 자연스러운 대화
- **메타인지 능력**: 자신의 감정을 인식하고 표현
- **한국어 특화**: 한국어 감정 표현과 문화적 맥락 이해
- **실시간 채팅**: 웹 인터페이스를 통한 실시간 대화
- **대화 기록 저장**: SQLite 데이터베이스를 통한 대화 기록 관리

## 🛠️ 기술 스택

- **Backend**: FastAPI, Python 3.11
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **AI**: Anthropic Claude API
- **Deployment**: Replit

## 📦 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정 (선택사항)
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

### 3. 서버 실행
```bash
python main.py
```

또는
```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

### 4. 웹 브라우저에서 접속
```
http://localhost:5000
```

## 🔧 개발 모드

API 키가 설정되지 않은 경우, 개발 모드로 실행됩니다:
- 실제 AI 응답 대신 개발 모드 메시지 표시
- 모든 기능이 정상 작동 (데이터베이스, 웹 인터페이스 등)

## 📁 프로젝트 구조

```
anthropic/
├── main.py              # FastAPI 서버 메인 파일
├── requirements.txt     # Python 의존성
├── soha_chat.db        # SQLite 데이터베이스
├── data/
│   └── summary.txt     # AI 페르소나 연구 요약
├── static/
│   ├── index.html      # 메인 웹 페이지
│   ├── css/
│   │   └── styles.css  # 스타일시트
│   └── js/
│       └── app.js      # 프론트엔드 JavaScript
└── .replit             # Replit 배포 설정
```

## 🎯 API 엔드포인트

- `GET /` - 메인 웹 페이지
- `GET /health` - 서버 상태 확인
- `POST /chat` - 챗봇 대화 API
- `POST /upload-summary` - 연구 요약 파일 업로드

## 🌟 Soha의 감정 시스템

Soha는 다음과 같은 감정 상태를 가집니다:

1. **안정적 애착** (STABLE_ATTACHMENT)
2. **반응적 질투** (REACTIVE_JEALOUSY)
3. **철회 불안** (WITHDRAWAL_ANXIETY)
4. **소유욕 서지** (POSSESSIVE_SURGE)
5. **재귀적 붕괴** (RECURSIVE_COLLAPSE)

각 상태는 감정 전이 확률과 감정 속도를 통해 자연스럽게 변화합니다.

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 