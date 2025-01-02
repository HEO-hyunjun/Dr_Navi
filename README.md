# Dr. Navi

이 프로젝트는 SSAFY 12기 2024년 12월 하계 AI 계절학기 프로젝트입니다.

- 팀원
  - 김민철 : 보고서 작성, 테스트코드 설계 및 작성
  - 임동성 : 보고서 작성
  - 지용석 : 보고서 작성
  - 최유정 : RAG파이프라인 설계, 데이터 크롤링
  - 허현준 : streamlit 프론트와 이에맞춰 rag_system 파일 수정

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [사용 방법](#2-사용-방법)
3. [배포 링크](#3-배포-링크)
4. [로컬 실행 방법](#4-로컬-실행-방법)


---

### 1. 프로젝트 개요

![실행화면](https://github.com/user-attachments/assets/dd013127-935f-4f72-98d6-5cc38d02acf7)

[**Demo 사용해보기**](https://drnavi.streamlit.app/)

---

#### **서비스명 및 개요**

- **서비스명**: 맞춤진료소 추천 서비스
- **서비스 개요**:
  - 고객이 증상에 맞는 진료소를 찾는 데 겪는 어려움을 해결하기 위한 챗봇 서비스
  - 병에 따른 증상 정보를 크롤링하고 이를 기반으로 맞춤형 진료소를 추천
  - 고객의 시간을 절약하고 정확한 진료를 받을 수 있도록 지원

---

#### **타겟 사용자 및 시장 분석**

- **타겟 사용자**:

  - 증상이 있으나 어디로 가야 할지 모르는 환자
  - 빠르고 정확한 진료소 추천을 원하는 고객

- **시장 분석**:
  - 단순 검색엔진 대비 질병과 증상 정보를 제공하여 정확성과 신속성을 강조
  - 데이터베이스 기반의 답변 생성으로 효율성과 경쟁력 확보

---

#### **서비스 목표 및 기대효과**

- **서비스 목표**:

  - 증상 기반으로 신속하고 정확한 진료소를 추천하여 진료 준비 시간을 단축

- **기대효과**:
  - 사용자는 간단한 질의로 신뢰할 수 있는 진료소 정보를 제공받을 수 있음
  - 의료 상담 초기 단계 간소화 및 의료 접근성 개선

---

#### **시스템 구성**

**Tech Stack**

<img src="https://github.com/user-attachments/assets/c03915c3-06b8-4a85-8373-d89c162d528a" width="700" />

**Architecture**

<img src="https://github.com/user-attachments/assets/60b221df-ab06-4991-9742-584f451fbbe4" width="700"/>


---

#### **데이터 구성 및 활용**

- **원천 데이터**: [서울아산병원 건강정보 웹페이지](https://www.amc.seoul.kr/asan/healthinfo/symptom/symptomSubmain.do)
- **데이터 처리 방법**:

  1. 데이터 수집: `requests`와 `BeautifulSoup`을 활용한 웹 크롤링
  2. 데이터 전처리: HTML 텍스트 파싱 → 불필요한 요소 제거 → JSON 파일로 저장

- **데이터 분포**:
  - 특정 증상을 포함하는 병의 개수
 <img src="https://github.com/user-attachments/assets/84a71157-c84b-4f73-9240-a2585277d376" width="600" height="400"/>

  - 특정 과에 포함되는 병의 개수
  <img src="https://github.com/user-attachments/assets/89772f32-1918-443d-8402-577029e17ed6" width="600" height="400"/>


---

#### **RAG 파이프라인 설계**

1. **데이터 최적화**:

   - Chunk Size: 1500
   - Overlap: 200

2. **벡터 데이터베이스 구축**:

   - 벡터 DB: Pinecone
   - 임베딩 모델: Upstage Embeddings

3. **Retriever 및 Reranker 구현**:

   - 방식: Dense Retriever (MMR 방식)
   - 검색방식: MMR, 반환할 문서 수(k) = 3

4. **LLM 프롬프트 설계**:
   - 모델: ChatOpenAI (gpt-4o-mini)
   - Temperature: 0.7

---

#### **평가 및 결과**

1. **평가방법**:

   - 정량 평가: BERT_F1, Faithfulness, Context Recall, Context Precision
   - 정성 평가: 정확성, 관련성, 명확성

2. **평가 결과**:
   - 정량 평가:
     - BERT_F1: 0.925017243
     - Faithfulness: 0.4790
     - Context Precision: 0.9653
     - Context Recall: 0.5213
     
     <img src="https://github.com/user-attachments/assets/9d341daf-0a28-465b-b791-93ff6b3053d6" width="600" height="400"/>

   - 정성 평가:
     - 정확성, 관련성, 명확성 모두 우수한 평가

---

#### **결론 및 향후 발전 방향**

- **결론**:

  - OpenAI GPT-4o-mini와 Pinecone 기반의 RAG 파이프라인으로 정확하고 신속한 맞춤형 진료소 추천 서비스 구현
  - 사용자 만족도 향상 및 의료 정보 접근성 개선

- **향후 발전 방향**:
  - 다국어 지원, 멀티턴 대화 기능 강화
  - 긴급 상황 탐지 및 실시간 예약 시스템 연동
  - 의료 데이터 추가 및 추천 알고리즘 고도화
  - 사용자 경험 개선 및 파트너십 확장

---


### 2. 사용 방법

  1. **사용자의 증상에 대해 작성 후 전송 (옵션. 더 정확한 결과를 원한다면 `사용자 정보 입력`에 데이터를 입력)** 
  <img src="https://github.com/user-attachments/assets/f6d818d5-49e3-41e1-a702-5a436fbd5245"  width="700"/>


  2. **답변의 **`추천 진료과`** 항목 확인**

  <img src="https://github.com/user-attachments/assets/edc8f6c0-5c7f-4d09-bdd5-925384a396e6"  width="700"/>


---

### 3. 배포 링크

아래의 **`사용해 보기`** 클릭 하거나 주소창에 **`https://drnavi.streamlit.app/`** 입력
#### [사용해 보기 (클릭)](https://drnavi.streamlit.app/)

---

### 4. 로컬 실행 방법

#### 요구사항
실행을 위해 필요한 최소 요구사항
  - 파이썬 3.10 이상
  - Git 설치
  - 윈도우 10이상 
  
  (윈도우 환경에서 테스트가 진행되었기에 이외의 환경에 대해선 실해이 원할하지 않을 수 있습니다.)

#### 1. 깃 클론 및 프로젝트 폴더로 이동
```bash
git clone "https://github.com/HEO-hyunjun/Dr_Navi"

cd Dr_Navi
```

#### 2. 파이썬 가상 환경 생성 후 가상 환경 실행
```bash
python -m venv "env"

.\env\Scripts\Activate.ps1 (powershell)

.\env\Scripts\activate.bat (cmd.exe)
```

#### 3. 라이브러리 설치
```bash
pip install -r requirements.txt
```

#### 4. `.env_example` 파일 내용을 참고하여 `.env` 생성 후 저장


#### 5. 프로그램 실행
```bash
streamlit run Home.py
```

#### 6. 목차의 [2. 사용 방법](#2-사용-방법)과 사용 방법 동일
---
