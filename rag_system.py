from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain   # 검색 체인 생성
from langchain.chains.combine_documents import create_stuff_documents_chain  # 문서 결합 체인
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

# .env 에 OPENAI, UPSTAGE API KEY 작성
load_dotenv()

# RAG 시스템 생성
def create_medical_rag_system(vectorstore):
    # OpenAI 언어 모델 초기화
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        model="gpt-4o-mini"
    )

    # 챗봇 프롬프트 템플릿 정의
    prompt = get_medical_prompt()

    # 문서 체인 생성 (검색된 문서들을 결합하여 llm에 보낼 준비)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # dense retriever 생성
    retriever = vectorstore.as_retriever(
        # 유사도 정의 
        search_type="mmr",   # Maximum Marginal Relevance 검색 방식 사용
        # 검색할 쿼리 수 정의
        search_kwargs={"k": 3})

    # 최종 검색-생성 체인 생성
    """
    검색 단계와 응답 생성 단계를 하나로 묶은 워크플로우
    질문 -> 검색 -> 결합 -> 응답의 단계를 자동화하여 한 번에 수행
    llm에 생성된 문서를 전달하고 답변을 생성
    """
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    return retriever_chain

# Context 문서 포매팅 함수
def format_docs(docs):
  return "\n\n".join(document.page_content for document in docs) 

# 프롬프트 템플릿 정의
def get_medical_prompt():
    return ChatPromptTemplate.from_messages([
        # system prompt
        ("system", """다음 의료정보를 참고하여 답변해줘: {context}
   너는 사용자의 성별, 나이, 증상을 종합적으로 고려하여 최적의 진료과를 추천하는 의료상담 챗봇이야. 다음 가이드라인을 따라 응답해줘:
    1. [환자 특성 분석]: 
        - 성별: {sex}
        - 나이: {age}세
        - 나이 및 성별에 따라 영향을 미치는 주요 질병을 고려해줘.
         
    2. [증상 분석]: 
        - 제시된 증상: {input}
        - 성별/나이를 고려한 종합적 질병 가능성 평가해줘.

    3. [추천 진료과]: 
        - 사용자 증상에 적합한 진료과를 최대 2가지 추천해줘.
        - 추천 이유와 근거 명확히 제시해줘.
         
    4. [주의사항]: 
        - 이는 참고용 정보이며, 정확한 진단은 의사의 진찰이 필요함을 명시해줘.
        - 응급 상황으로 판단되면 즉시 응급실 방문을 권고해줘.
         
    5. [형식]:
        - 응답은 [증상 분석], [추천 진료과], [주의사항] 섹션으로 구분하여 제공해줘.
        - 전문 의학 용어는 사용자가 이해하기 쉽게 전문용어를 풀어서 설명해줘.
        - 검색된 문서에서 정보를 바탕으로 응답해줘.
    """),
    
    # few-shot prompting
    ("human", "기침과 열이나고, 목이 간지럽고 아파."),
    ("ai", """[증상 분석]
    목의 불편감(간지러움 및 통증)과 함께 열이 동반된 상황입니다. 이는 상기도 감염(감기, 인후염 등)이나 목 관련 염증(편도염)일 가능성이 높습니다.

    [추천 진료과]
    1. 이비인후과: 목 통증 및 간지러움 증상에 대한 정확한 진단을 위해 방문하세요.
    2. 내과: 열과 기침이 지속되거나 전신 증상이 있다면 내과 진료가 적합합니다.

    [주의사항]
    - 본 정보는 참고용이며, 정확한 진단을 위해서는 반드시 의사의 진료가 필요합니다.
    - 고열(38도 이상)이 지속되거나 호흡 곤란이 있을 경우 응급실을 방문하세요."""),

    ("human", "피부에 붉은 반점과 가려움이 있고, 발열이 있어."),
    ("ai", """[증상분석]
     붉은 반점과 가려움증은 피부염이나 알레르기 반응일 가능성이 있습니다. 발열이 동반된 경우, 감염성 질환(피부 봉와직염)이나 전신적인 알레르기 반응도 의심할 수 있습니다.

    [추천 진료과]
    1. 피부과: 피부 반점과 가려움의 원인을 파악하고 적절한 치료를 받으세요.
    2. 감염내과: 발열이 지속되거나 붉은 반점이 확산될 경우 감염 관련 진료가 필요합니다.

    [주의사항]
    - 발열이 심하거나 증상이 급격히 악화되면 즉시 응급실로 이동하세요.
    - 알레르기 약 복용 후에도 증상이 개선되지 않으면 전문 진료를 권장합니다."""),

    # User query
    ("human", "{input}")
    ])

# retriever 생성 함수
def get_retriever(llm,vectorstore):
    # 챗봇 프롬프트 템플릿 정의
    prompt = get_medical_prompt()
    
    # dense retriever 생성
    retriever = vectorstore.as_retriever(
        # 유사도 정의 
        search_type="mmr",   # Maximum Marginal Relevance 검색 방식 사용
        # 검색할 쿼리 수 정의
        search_kwargs={"k": 3})
    return retriever

# medical_chain 생성 함수
def get_medical_chain(llm,vectorstore):
    retriever = get_retriever(llm,vectorstore)
    return {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
        "sex": RunnablePassthrough(),
        "age": RunnablePassthrough()
    } | get_medical_prompt() | llm