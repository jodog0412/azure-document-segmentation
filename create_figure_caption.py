from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

import os
import typing as tp
import pymupdf
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

document_endpoint = os.getenv('DOCUMENT_INTELLIGENCE_ENDPOINT')
document_api_key = os.getenv('DOCUMENT_INTELLIGENCE_API_KEY')
vision_endpoint = os.getenv('VISION_ENDPOINT')
vision_api_key = os.getenv('VISION_KEY')

INPUT_DIR_PATH = 'input'  
OUTPUT_DIR_PATH = 'output'
# helper functions
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
        
def recognize_figs_from_page(page_path: str) -> tp.List[tp.List[int]]:
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=document_endpoint, 
        credential=AzureKeyCredential(document_api_key)
    )

    with open(page_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", 
            analyze_request=f, 
            features=[],  # Specify which add-on capabilities to enable
            content_type="application/octet-stream"
        )
    result: AnalyzeResult = poller.result()

    bboxes = []
    if result.figures:                    
        for figures_idx, figures in enumerate(result.figures):
            for region in figures.bounding_regions:
                print(f"Figure # {figures_idx} location on page:{region.page_number} is within bounding polygon '{region.polygon}'")  
                bboxes.append(region.polygon)
    return bboxes

def extract_figs(page_path: str, bboxes: tp.List[tp.List[int]]) -> list:
    page_name = page_path.split('/')[-1].replace('.jpg','')
    page = Image.open(page_path)
    figs = []
    for idx, bbox in enumerate(bboxes):
        xs, ys = bbox[::2], bbox[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        fig = page.crop((x1, y1, x2, y2))
        fig_path = f'{OUTPUT_DIR_PATH}/{page_name}_fig{idx}.jpg'
        figs.append(fig_path)
        fig.save(fig_path)
    return figs

def create_fig_caption(fig_path: str) -> str:
    with Image.open(fig_path) as fig:
        w, h = fig.size
        if w<50 or w>16000 or h<50 or h>16000:
            return "Invaild image size. Size of image is too small or big."
        
    client = ImageAnalysisClient(
        endpoint=vision_endpoint,
        credential=AzureKeyCredential(vision_api_key)
    )

    # Load image to analyze into a 'bytes' object
    with open(fig_path, "rb") as f:
        fig = f.read()

    # Get a caption for the image. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=fig,
        visual_features=[VisualFeatures.CAPTION],
        gender_neutral_caption=True,  # Optional (default is False)
    )

    return result.caption.text

file_name = 'edu_01.pdf'
local_doc = f'{INPUT_DIR_PATH}/{file_name}'
pages = split_pages_from_document(local_doc)
for page in pages:
    bboxes = recognize_figs_from_page(page)
    figs = extract_figs(page, bboxes)
    for fig in figs:
        print(create_fig_caption(fig))
