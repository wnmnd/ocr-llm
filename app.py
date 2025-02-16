from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import easyocr
import keras_ocr
from paddleocr import PaddleOCR
import pytesseract  # Added Tesseract OCR
from PIL import Image
import io

app = FastAPI()

# OCR Functions with Confidence Scores
def ocr_with_paddle(img):
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    result = ocr.ocr(img)

    extracted_text = []
    confidences = []

    for line in result[0]:  # PaddleOCR returns nested lists
        text, confidence = line[1]
        extracted_text.append(text)
        confidences.append(confidence)

    return extracted_text, confidences

def ocr_with_keras(img):
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(img)]
    predictions = pipeline.recognize(images)

    extracted_text = [text for text, confidence in predictions[0]]
    confidences = [confidence for text, confidence in predictions[0]]

    return extracted_text, confidences

def ocr_with_easy(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray_image)

    extracted_text = [text for _, text, confidence in results]
    confidences = [confidence for _, text, confidence in results]

    return extracted_text, confidences

def ocr_with_tesseract(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image).split("\n")

    # Tesseract does not return confidence scores by default
    confidences = [1.0] * len(extracted_text)  

    return extracted_text, confidences

# API Endpoint
@app.post("/extract-text/")
async def extract_text(ocr_method: str, file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read()))
        img_cv = np.array(image)

        # Select OCR method
        if ocr_method == "PaddleOCR":
            extracted_text, confidences = ocr_with_paddle(img_cv)
        elif ocr_method == "EasyOCR":
            extracted_text, confidences = ocr_with_easy(img_cv)
        elif ocr_method == "KerasOCR":
            extracted_text, confidences = ocr_with_keras(img_cv)
        elif ocr_method == "TesseractOCR":
            extracted_text, confidences = ocr_with_tesseract(img_cv)
        else:
            raise HTTPException(status_code=400, detail="Invalid OCR method! Choose PaddleOCR, EasyOCR, KerasOCR, or TesseractOCR.")

        # Clean and structure the response
        response_data = {
            "ocr_method": ocr_method,
            "extracted_text": extracted_text,
            "confidence_scores": confidences
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server with: uvicorn app:app --host 0.0.0.0 --port 8000
