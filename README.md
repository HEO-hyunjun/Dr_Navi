# Dr. Navi

이 프로젝트는 SSAFY 12기 2024년 12월 하계 AI 계절학기 프로젝트입니다.

- 팀원
  - 김민철 : 보고서 작성, 테스트코드 설계 및 작성
  - 임동성 : 보고서 작성
  - 지용석 : 보고서 작성
  - 최유정 : RAG파이프라인 설계, 데이터 크롤링
  - 허현준 : streamlit 프론트와 이에맞춰 rag_system 파일 수정

---

![실행화면](https://github.com/user-attachments/assets/d866d614-df59-4607-a34d-5cf303b92131)

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
   - 온도(Temperature): 0.7

---

#### **평가 및 결과**

1. **평가방법**:

   - 정량 평가: Context Precision, Recall, Faithfulness, Answer Relevancy
   - 정성 평가: 정확성, 관련성, 명확성

2. **평가 결과**:
   - 정량 평가:
     - Context Precision: 0.5833
     - Context Recall: 0.6667
     - Faithfulness: 0.8333
     - Answer Relevancy: 0.7112
     <img src="https://github.com/user-attachments/assets/206fb8d2-7c8d-4657-8841-eec9ede1bdcb" width="600" height="400"/>

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
