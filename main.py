from crawl_data import get_vectorstore
from rag_system import create_medical_rag_system

############### python main.py 실행  ###################
## 콘솔 테스트용
def medical_info():
    # 성별별
    while True:
        sex = input("성별을 입력해주세요(남성/여성, 중단은 'q' 입력): ").strip()
        if sex.lower() == 'q':
            return None 
        if sex in ["남성", "여성"]:
            break
        print("남성 또는 여성으로 입력해주세요.")

    # 나이
    while True:
        age_input = input("나이를 입력해주세요(중단은 'q' 입력): ").strip()
        if age_input.lower() == 'q':
            return None  
 
        age = int(age_input)
        if 0 < age < 120:
            break
        print("올바른 나이를 입력해주세요.")

    # 증상
    while True:
        symptoms = input("증상을 입력해주세요(중단은 'q' 입력): ").strip()
        if symptoms.lower() == 'q':
            return None  
        if symptoms:
            break
        print("증상을 입력해주세요.")

    # 수집된 정보 딕셔너리로 반환
    return {
        "sex": sex,
        "age": age,
        "symptoms": symptoms
    }


def main():
    vectorstore = get_vectorstore()
    
    # RAG 시스템 생성
    medical_chain = create_medical_rag_system(vectorstore)
    
   # 대화형 인터페이스
    while True:
        # 정보 수집
        user_info = medical_info()

        response = medical_chain.invoke({
            "input": user_info["symptoms"],
            "sex": user_info["sex"],
            "age": user_info["age"]
        })
        print("\n=== 답변 ===")
        print(response["answer"])

if __name__ == "__main__":
    main()