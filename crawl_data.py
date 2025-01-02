from bs4 import BeautifulSoup 
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
import requests, os, json

load_dotenv()

def crawl_disease_symptom():
    documents = []
    seen_diseases = set()  # 중복제거
    
    url = "https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseList.do?diseaseKindId=C00000{}".format(20)
    response = requests.get(url)  # GET 메소드로 url에 HTTP Requset 전송
    print(f"HTTP 응답 상태 코드: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')  # 응답받은 html 파싱

    # 각 질환별로 들어가기 
    for link in soup.select('div.tabSearchList.cont2')[0].find_all('a'):
        # print(link)
        link_code = link['href'][-6:]   # url의 마지막 6자리 코드 추출
        link_url = "https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseList.do?diseaseKindId=C{}".format(link_code)
        response = requests.get(link_url)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # 페이지네이션 확인
        pagination = soup.select('div.pagingWrapSec > span > a[onclick]')
        total_pages = int(pagination[-1].get_text() if pagination else 1)

        for page in range(1, total_pages + 1):
            page_url = f"{link_url}&pageIndex={page}" if total_pages > 1 else link_url   # 페이지가 하나이면 원래 link
            response = requests.get(page_url)
            page_soup = BeautifulSoup(response.content, 'html.parser')
            disease_elements = page_soup.select('ul.descBox > li > div.contBox')  # 질병 정보 
        
            for disease_element in disease_elements:
                disease_name = disease_element.find('strong').get_text(strip=True)  # 질병명
            
                # 중복제거
                if disease_name in seen_diseases:
                    continue
                seen_diseases.add(disease_name)

                # 증상 추출 (중복제거)
                symptoms = set()
                departments = set()
                links = disease_element.select('dd a')
                for link in links:  # 세부정보
                    # symptomId가 있는 링크만 추출
                    if 'symptomId' in link['href']:  
                        symptoms.add(link.get_text(strip=True))
                    # dept가 있는 링크
                    elif 'dept' in link['href']:
                        departments.add(link.get_text(strip=True))

                # 증상과 진료과가 모두 있는 경우만 저장
                # 질병-증상 데이터를 RAG용 문서(Document 객체)로 변환
                if symptoms and departments:
                    documents.append(Document(
                        page_content=f"""질병명: {disease_name}
                        주요증상: {', '.join(symptoms)}
                        진료과: {', '.join(departments)}""",
                        metadata={
                            "disease": disease_name,
                            "symptoms": ', '.join(symptoms),
                            "departments": ', '.join(departments) 
                            }
                        ))
    # print(documents)
    return documents


# Vectorsotre
def get_vectorstore():
    """
    # Chroma DB 경로 설정
    persist_directory = "./chroma_db"

    # OPENAIEmbeddings 모델 사용 불가
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-passage"
    )

     # Chroma DB 중복방지 위해 데이터 확인 
    if os.path.exists(persist_directory):
        print("기존 db 사용")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print("새로운 db 생성")
        # 문서 준비 및 Vector Store 생성
        documents = crawl_disease_symptom()
        return Chroma.from_documents(
        documents=documents,
        embedding=embeddings, 
        persist_directory=persist_directory
        )

    """
    index_name = 'medical-chatbot'

    # pinecone 초기화 
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 인덱스 삭제
    # if index_name in pc.list_indexes().names():
    #     pc.delete_index(index_name)

    # 임베딩 모델 초기화 
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-query"
    )

    # 테스트용 임베딩으로 차원 확인
    test_embedding = embeddings.embed_query("test")
    dimension = len(test_embedding)

    # pinecone index가 없다면 생성
    if index_name not in pc.list_indexes().names():
        print("새로운 인덱스 생성")
        pc.create_index(
            name=index_name,
            dimension=dimension,   # 벡터 차원 수
            metric="cosine",  # 벡터간 유사도 계산 방법  
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # 서버리스 사양 설정
        )

        # vecotrstore 생성
        pinecone_vectorstore = PineconeVectorStore.from_documents(
            documents=crawl_disease_symptom(),
            index_name=index_name,
            embedding=embeddings
        )
    else:
        pinecone_vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        print("기존 데이터 사용")

    return pinecone_vectorstore
