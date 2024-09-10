from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature

import os
import typing as tp
import pymupdf
import re
from dotenv import load_dotenv
load_dotenv()

document_endpoint = os.getenv('DOCUMENT_INTELLIGENCE_ENDPOINT')
document_api_key = os.getenv('DOCUMENT_INTELLIGENCE_API_KEY')
vision_endpoint = os.getenv('VISION_ENDPOINT')
vision_api_key = os.getenv('VISION_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

INPUT_DIR_PATH = 'input'  
OUTPUT_DIR_PATH = 'output'
# helper functions

def is_number(text):
    # 정수 또는 실수를 판단할 수 있는 정규식 패턴
    pattern = r'^-?\d+(\.\d+)?$'
    # 패턴에 맞는지 확인
    return re.match(pattern, text) is not None

def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result

def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def text_preprocessing(text:str) -> str:
    if ':formula:' in text:
        if text == ':formula:':
            return ""
        else:
            text = text.replace(':formula:','')
    if is_number(text):
        return ""
    return text

def text_postprocessing(text:str) -> str:
    client = OpenAI(api_key=openai_api_key)

    query_example = "주어진 글에 대해 줄바꿈을 해서 반환해줘.\
        줄바꿈 규칙은 다음과 같아.\
        1. 문장이 마무리되면 줄바꿈한다.\
        2. 문장이 쉼표로 마무리되면 줄바꿈한다.\
        3. 문장에 2개 이상의 띄어쓰기는 절대로 존재해서는 안된다.\
        \
        주어진 글은 다음과 같아.\
        STEP 1 | 유형별 문제 공략 하기 다음을 만족하는 단항식 A, B에 대하여  하여라. a, b의 값을 각각 구하여라. 단항식의 곱셈과 나눗셈의 응용 문제에서 주어진 조건에 맞게 식을 세운 후 문제가 요구 참고 J 때 이 물통의 높이를 구하여라. 밑면의 반지름의 길이가  4r인 원기둥 모양 의 통에 음료수가 가득 차 있다. 이 음료수를 반지름의 길 이가 그가인 반구 모양의 컵에 가득 담아 사람들에게 나누 어 주려고 할 때, 최대 몇 명의 사람들에게 음료수를 나누 어 줄 수 있는지 구하여라.\
        (단, 통과 컵의  다음 그림과 같이 서로 합동인 25개의 작은 직사각형으로 이루 어진 큰 직사각형의 가로의 길이와 세로의 길이는 각 각 10ab3,  . 이때 검은 직사각형의 넓이의 합과 흰 직사각형의 넓이의 합을 순서대로 구하여라. 10ab3 다음 그림과 같이 밑면의 반지름의 길이가 2a이고, 높이  원기둥 안에 크기가 같은 2개의 구가 꼭 맞게 들 어 있다."
    
    response_example = "STEP 1 | 유형별 문제 공략 하기\
        다음을 만족하는 단항식 A, B에 대하여  하여라.\
        a, b의 값을 각각 구하여라.\
        단항식의 곱셈과 나눗셈의 응용 문제에서 주어진 조건에 맞게 식을 세운 후 문제가 요구 참고 J때 이 물통의 높이를 구하여라.\
        밑면의 반지름의 길이가  4r인 원기둥 모양 의 통에 음료수가 가득 차 있다.\
        이 음료수를 반지름의 길 이가 그가인 반구 모양의 컵에 가득 담아 사람들에게 나누 어 주려고 할때, 최대 몇 명의 사람들에게 음료수를 나누 어 줄 수 있는지 구하여라. (단, 통과 컵의\
        다음 그림과 같이 서로 합동인 25개의 작은 직사각형으로 이루어진 큰 직사각형의 가로의 길이와 세로의 길이는 각 각  10ab3,  .\
        이때 검은 직사각형의 넓이의 합과 흰 직사각형의 넓이의 합을 순서대로 구하여라. 10ab3\
        다음 그림과 같이 밑면의 반지름의 길이가 2a이고, 높이  원기둥 안에 크기가 같은 2개의 구가 꼭 맞게 들어 있다."
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query_example},
            {"role": "assistant", "content": response_example},
            {"role": "user", 
             "content": f"주어진 글에 대해 줄바꿈('\n')을 하여 반환해줘.\
                줄바꿈 규칙은 다음과 같아.\
                1. 문장이 마무리되면 줄바꿈한다.\
                2. 문장이 쉼표로 마무리되면 줄바꿈한다.\
                3. 문장에 2개 이상의 띄어쓰기는 절대로 존재해서는 안된다.\
                주어진 글은 다음과 같아.\
                {text}"},
        ]
    )
    return completion.choices[0].message.content

def split_pages_from_document(doc_path: str) -> tp.List[str]:
    doc_name = doc_path.split('/')[-1].replace('.pdf','')
    doc = pymupdf.open(doc_path)
    pages = []
    for idx in range(len(doc)):
        page = doc[idx]
        pix = page.get_pixmap(dpi=450)

        page_path = f'{OUTPUT_DIR_PATH}/{doc_name}_P{idx}.jpg'
        pages.append(page_path)
        pix.save(page_path)
    return pages
        
def recognize_texts_from_page(page_path: str) -> tp.List[str]:
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=document_endpoint, 
        credential=AzureKeyCredential(document_api_key)
    )

    with open(page_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", 
            analyze_request=f, 
            features=[DocumentAnalysisFeature.FORMULAS],  # Specify which add-on capabilities to enable
            content_type="application/octet-stream"
        )
    result: AnalyzeResult = poller.result()

    for page in result.pages:
        print(f"----Analyzing layout from page #{page.page_number}----")
        print(f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}")

        lines = []
        # Analyze lines.
        if page.lines:
            for line_idx, line in enumerate(page.lines):
                text = line.content
                text = text_preprocessing(text)
                lines.append(text)

    return lines

file_name = 'edu_02.pdf'
doc_path = f'{INPUT_DIR_PATH}/{file_name}'
pages = split_pages_from_document(doc_path)
for page in pages:
    text_group = recognize_texts_from_page(page)
    text = ' '.join(text_group)
    text = text_postprocessing(text)
    print(text)