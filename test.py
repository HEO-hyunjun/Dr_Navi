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

question_list = [
    "기침, 가래, 호흡곤란",
    "기침, 가래, 열, 빈호흡",
    "심부전, 호흡곤란",
    "빈뇨, 야간뇨, 방광 팽만",
    "환부통증",
    "청력장애, 이명",
    "발적, 청력장애, 고막에 수포형성",
    "어깨통증, 눈의 출혈",
    "환부의 분비물, 눈의 통증",
    "권태감, 근력 약화",
    "요통, 골반통",
    "환부 통증, 덩어리가 만져짐"
]

ground_truth_list = [
    "호흡기내과, 내과",
    "감염내과",
    "호흡기내과",
    "비뇨의학과",
    "정형외과",
    "이비인후과",
    "이비인후과",
    "정형외과",
    "안과",
    "신경과",
    "재활의학과, 류마타내스내과",
    "성형외과, 정형외과"
]

contexts_list = []
answers_list = []

def test():
    vectorstore = get_vectorstore()

    # RAG 시스템 생성
    medical_chain = create_medical_rag_system(vectorstore)

    contexts_list = []  # Initialize lists
    answers_list = []

    for question in question_list:
        response = medical_chain.invoke({"input": question})

        docs = response['context']

        # Collect page contents as a list
        combined_content = [doc.page_content for doc in docs]
        print("\n".join(combined_content))  # For debugging

        contexts_list.append(combined_content)  # Store as list
        answers_list.append(parse(response["answer"]))
    
    df = pd.DataFrame(
        {
            "question": question_list,
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