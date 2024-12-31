from crawl_data import get_vectorstore
from rag_system import create_medical_rag_system

############### python main.py 실행  ###################
## 콘솔 테스트용

def main():
    vectorstore = get_vectorstore()
    
    # RAG 시스템 생성
    medical_chain = create_medical_rag_system(vectorstore)
    
    # 대화형 인터페이스
    while True:
        user_input = input("\n증상을 입력하세요 (종료는 'q'): ")
        if user_input.lower() == 'q':
            break
    
        response = medical_chain.invoke({"input": user_input})
        print("\n=== 답변 ===")
        print(response["answer"])

if __name__ == "__main__":
    main()