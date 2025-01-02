from langchain.schema import BaseOutputParser
from crawl_data import get_vectorstore
from rag_system import create_medical_rag_system
from langchain_openai import ChatOpenAI
from datasets import Dataset
import pandas as pd
import re

from ragas import evaluate

def parse(text):
        # 1. [추천 진료과] 항목을 찾는다
        match = re.search(r'\[추천 진료과\](.*?)\[', text, re.DOTALL)
        if match:
            recommendations_section = match.group(1)
        else:
            return []

        # 2. ~과 로 끝나는 단어를 리스트로 반환한다
        recommendations = re.findall(r'\b\S+과\b', recommendations_section)
        return ", ".join(recommendations)


test_cases = [
    {
        "symptoms": "기침, 가래, 호흡곤란",
        "sex": "남성", 
        "age": 45,
        "ground_truth": "호흡기내과, 내과"
    },
    {
        "symptoms": "기침, 가래, 열, 빈호흡",
        "sex": "여성", 
        "age": 35,
        "ground_truth": "감염내과"
    },
    {
        "symptoms": "어깨통증, 손 저림, 두통, 팔 저림",
        "sex": "남성", 
        "age": 45,
        "ground_truth": "신경외과, 정형외과"
    },
    {
        "symptoms": "복부통증, 골반 통증",
        "sex": "여성", 
        "age": 30,
        "ground_truth": "산부인과"
    },
        {
        "symptoms": "유방멍울",
        "sex": "남성", 
        "age": 40,
        "ground_truth": "유방외과"
    },
        {
        "symptoms": "시야장애, 발작, 팔다리 마비",
        "sex": "남성", 
        "age": 5,
        "ground_truth": "신경과, 소아신경과, 신경외과"
    },
        {
        "symptoms": "배뇨곤란, 잔뇨감, 빈뇨",
        "sex": "남성", 
        "age": 70,
        "ground_truth": "비뇨의학과"
    },
        {
        "symptoms": "시야흐림, 복시, 눈부심",
        "sex": "여성", 
        "age": 70,
        "ground_truth": "안과"
    },
        {
        "symptoms": "어깨통증, 어깨운동 제한, 어깨 마찰음",
        "sex": "남성", 
        "age": 60,
        "ground_truth": "정형외과"
    },
        {
        "symptoms": "얼굴 홍반, 관절 통증, 피로감",
        "sex": "여성", 
        "age": 50,
        "ground_truth": "류마티스내과, 피부과"
    },
        {
        "symptoms": "다식, 다음, 다뇨, 체중감소소",
        "sex": "남성", 
        "age": 60,
        "ground_truth": "내분비내과"
    },
        {
        "symptoms": "귀 통증, 귀 분비물, 열",
        "sex": "여성", 
        "age": 10,
        "ground_truth": "이비인후과"
    },
]


contexts_list = []
answers_list = []

def test():
    vectorstore = get_vectorstore()

    # RAG 시스템 생성
    medical_chain = create_medical_rag_system(vectorstore)

    contexts_list = []  # Initialize lists
    answers_list = []
    sex_list = []
    age_list = []
    symptoms_list = []
    ground_truth_list = []


    for case in test_cases:
        response = medical_chain.invoke({
            "input": case["symptoms"],
            "sex": case["sex"], 
            "age": case["age"]
             })

        docs = response['context']

        # Collect page contents as a list
        combined_content = [doc.page_content for doc in docs]
        print(f"\n성별: {case['sex']}, 나이: {case['age']}, 증상: {case['symptoms']}")
        print("\n".join(combined_content))  # For debugging

        contexts_list.append(combined_content)  # Store as list
        answers_list.append(parse(response["answer"]))
        sex_list.append(case["sex"])
        age_list.append(case["age"])
        symptoms_list.append(case["symptoms"])
        ground_truth_list.append(case["ground_truth"])
    
    df = pd.DataFrame(
        {
            "question": symptoms_list,
            "sex": sex_list,
            "age": age_list,
            "contexts": contexts_list,  # Ensure it's a list
            "answer": answers_list,
            "ground_truth": ground_truth_list,  # Adjust if `ground_truth_list` contains multiple items
        }
    )

    rag_results = Dataset.from_pandas(df)

    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )

    result = evaluate(
        rag_results,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        ],
    )

    print(result)


if __name__ == "__main__":
    test()