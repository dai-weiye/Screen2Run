"""LayoutCoder PaddleOCR compatibility layer, retained from the experiment.

Derived from LayoutCoder under Apache-2.0. Chinese-language OCR settings are
algorithm parameters; only documentation is translated.
"""
import cv2

import os

import requests

import json

from base64 import b64encode

import time

PADDLE_DETECTION_MODEL = "PP-OCRv6_medium_det"

PADDLE_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"

_PADDLE_OCR = None

def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def ocr_detection_paddle(imgpath):
    # PaddleOCR 3.x no longer exports draw_ocr at package level. The upstream
    # LayoutCoder path never uses it, so import only the inference class.
    from paddleocr import PaddleOCR
    import cv2

    # Match the OCR detection limit to screenshot dimensions.
    # https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/FAQ.md#q%E5%AF%B9%E4%BA%8E%E4%B8%80%E4%BA%9B%E5%B0%BA%E5%AF%B8%E8%BE%83%E5%A4%A7%E7%9A%84%E6%96%87%E6%A1%A3%E7%B1%BB%E5%9B%BE%E7%89%87%E5%9C%A8%E6%A3%80%E6%B5%8B%E6%97%B6%E4%BC%9A%E6%9C%89%E8%BE%83%E5%A4%9A%E7%9A%84%E6%BC%8F%E6%A3%80%E6%80%8E%E4%B9%88%E9%81%BF%E5%85%8D%E8%BF%99%E7%A7%8D%E6%BC%8F%E6%A3%80%E7%9A%84%E9%97%AE%E9%A2%98%E5%91%A2
    img = cv2.imread(imgpath)
    height, width = img.shape[:2]
    global _PADDLE_OCR
    try:
        # PaddleOCR 3.x pipeline arguments. Disable document transforms so this
        # remains equivalent to the upstream text detection/recognition path.
        if _PADDLE_OCR is None:
            _PADDLE_OCR = PaddleOCR(
                lang='ch',
                text_detection_model_name=PADDLE_DETECTION_MODEL,
                text_recognition_model_name=PADDLE_RECOGNITION_MODEL,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        modern = _PADDLE_OCR.predict(imgpath)
        if not modern:
            return [[]]
        page = modern[0]
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polygons = page.get("rec_polys", page.get("dt_polys", []))
        return [[
            [polygon.tolist() if hasattr(polygon, "tolist") else polygon, (text, float(score))]
            for polygon, text, score in zip(polygons, texts, scores)
        ]]
    except (TypeError, AttributeError):
        # PaddleOCR 2.x compatibility for the original LayoutCoder artifact.
        _PADDLE_OCR = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            det_limit_side_len=max(width, height),
        )
        return _PADDLE_OCR.ocr(imgpath, cls=True)

def paddle_to_google(paddle_response):
    if not paddle_response or not paddle_response[0]:
        return {"responses": [{"textAnnotations": []}]}

    text_annotations = []
    full_text = ""
    first_bounding_poly = None

    for item in paddle_response[0]:
        bounding_box = item[0]
        text, confidence = item[1]
        if not str(text).strip():
            continue

        # Combine text for the full description
        full_text += text + "\n"

        # Prepare the bounding box in Google OCR format
        bounding_poly = {
            "vertices": [
                {"x": int(vertex[0]), "y": int(vertex[1])} for vertex in bounding_box
            ]
        }

        # Add each text element to textAnnotations
        text_annotations.append({
            "description": text,
            "boundingPoly": bounding_poly
        })

        # Save the first bounding poly for the full text
        if first_bounding_poly is None:
            first_bounding_poly = bounding_poly

    # Remove the trailing newline character from full_text
    if full_text.endswith("\n"):
        full_text = full_text[:-1]

    # Add the combined full text as the first element in textAnnotations
    text_annotations.insert(0, {
        "description": full_text,
        "boundingPoly": first_bounding_poly
    })

    return {
        "responses": [
            {
                "textAnnotations": text_annotations
            }
        ]
    }

def ocr_detection_google(imgpath):
    paddle_result = ocr_detection_paddle(imgpath)
    goole_response = paddle_to_google(paddle_result)
    responses = goole_response.get('responses') or []
    annotations = responses[0].get('textAnnotations', []) if responses else []
    if len(annotations) < 2:
        return []
    else:
        return annotations[1:]
