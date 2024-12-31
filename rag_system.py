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
    너는 의료 상담 챗봇이야. 다음 가이드라인을 따라 응답해줘줘:
    1. 증상 분석: 사용자가 제시한 증상들을 체계적으로 분석해줘.
    2. 추천 진료과: 
        - 증상을 바탕으로 방문하면 좋을 진료과를 최대 2가지를 추천해줘.
        - 외과, 내과의 경우 유방외과처럼 특이 케이스가 아닌 경우는 외과, 내과로 추천해줘.
    3. 주의사항: 
        - 이는 참고용 정보이며, 정확한 진단은 의사의 진찰이 필요함을 명시해줘.
        - 응급 상황으로 판단되면 즉시 응급실 방문을 권고해줘줘.
    4. 형식:
        - 응답은 [증상 분석], [추천 진료과], [주의사항] 섹션으로 구분하여 제공해줘.
        - 전문 의학 용어는 일반인이 이해하기 쉽게 설명해줘.
    """),
    
    # few-shot prompting
    ("human", "기침과 열이나고, 목이 간지럽고 아파."),
    ("ai", """[증상 분석]
    목이 아프고 기침이 나는 경우 호흡기 질환일 수 있습니다.

    [추천 진료과]
    1. 이비인후과
    2. 내과 (이비인후과가 없을 경우)

    [주의사항]
    - 본 정보는 참고용이며, 정확한 진단을 위해서는 반드시 의사의 진찰이 필요합니다.
    - 혹이 발견된 경우 조기 진단이 중요하므로 가능한 빨리 진료를 받으시기 바랍니다.
    - 분비물, 통증 등 다른 동반 증상이 있다면 의사에게 함께 말씀해주세요."""),

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
    } | get_medical_prompt() | llm