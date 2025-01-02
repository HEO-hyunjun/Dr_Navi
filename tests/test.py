from langchain.schema import BaseOutputParser
from src.crawl_data import get_vectorstore
from src.rag_system import create_medical_rag_system
from langchain_openai import ChatOpenAI
from datasets import Dataset
import pandas as pd
import json
import re
from ragas import evaluate


file_path = "test_data.json"
test_json = json.load(open(file_path, encoding="UTF-8"))
contexts_list = []
answers_list = []

def make_ground_json(symptom_analyze,recomendation1,recomendation2):
    text = f"[증상 분석] \n{symptom_analyze}\n\n[추천 진료과]\n1.{recomendation1}\n\n2.{recomendation2}\n\n[주의사항]\n- 이는 참고용 정보이며, 정확한 진단은 의사의 진찰이 필요합니다.\n- 만약 증상이 심해지거나 지속된다면 즉시 병원을 방문하는 것을 권장합니다."
    return text

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


    for case in test_json:
        response = medical_chain.invoke({
            "input": case["증상"],
            "sex": case["성별"],
            "age": case["나이"]
             })

        # print(response["answer"])
        docs = response['context']

        # Collect page contents as a list
        combined_content = [doc.page_content for doc in docs]
        # print(f"\n성별: {case['성별']}, 나이: {case['나이']}, 증상: {case['증상']}")
        # print("\n".join(combined_content))  # For debugging  combinelist 출력
        # print("-"*50)

        contexts_list.append(combined_content)  # Store as list
        answers_list.append(response["answer"])
        # print(answers_list[-1]) #
        # print("-"*50)
        sex_list.append(case["성별"])
        age_list.append(case["나이"])
        symptoms_list.append(case["증상"])

        ground_truth=make_ground_json(case["증상 분석"],case["추천 진료과 1"],case["추천 진료과 2"])
        # print(ground_truth)
        ground_truth_list.append(ground_truth)

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
    # print("-"*50)
    rag_results = Dataset.from_pandas(df)

    from ragas.metrics import (
        # answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )
    df.to_csv("test.csv")

    ## 있는 파일 확인하고 싶을때,
    # import ast
    # df = pd.read_csv("result/test.csv")
    # if 'contexts' in df.columns:
    #     df['contexts'] = df['contexts'].apply(
    #         lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x
    #     )
    # rag_results = Dataset.from_pandas(df)

    result = evaluate(
        rag_results,
        metrics=[
            # answer_relevancy, # 답의 질문과의 관련성
            faithfulness, # 질문이 context로 부터 추출한 정보의 수
            context_recall, # context로 유추할 수 있는 답변 비율
            context_precision, # 질문에 관련있는 문서 잘 불러오는가.
        ],
    )

    print(result)


if __name__ == "__main__":
    test()