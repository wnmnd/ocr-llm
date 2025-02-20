from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import easyocr
import keras_ocr
from paddleocr import PaddleOCR
import pytesseract 
from PIL import Image
import io
import json
import os

app = FastAPI()
RESULTS_FILE = "ocr_results.json"

# Function to load existing results
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as file:
            return json.load(file)
    return []

# Function to save results
def save_results(data):
    with open(RESULTS_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Image Resizing Function (Maintains Aspect Ratio)
def image_resize(image, target_width=1024):
    """ Resizes an image while maintaining aspect ratio. """
    h, w = image.shape[:2]
    scale_ratio = target_width / w
    target_height = int(h * scale_ratio)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized_image

# Updated OCR functions to use resized images
def ocr_with_paddle(img):
    resized_img = image_resize(img)
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    result = ocr.ocr(resized_img)
    extracted_text = []
    confidences = []
    for line in result[0]:
        text, confidence = line[1]
        extracted_text.append(text)
        confidences.append(confidence)
    return extracted_text, confidences

def ocr_with_keras(img):
    resized_img = image_resize(img)
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(resized_img)]
    predictions = pipeline.recognize(images)
    extracted_text = [text for text, confidence in predictions[0]]
    confidences = [confidence for text, confidence in predictions[0]]
    return extracted_text, confidences

def ocr_with_easy(img):
    resized_img = image_resize(img)
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray_image)
    extracted_text = [text for _, text, confidence in results]
    confidences = [confidence for _, text, confidence in results]
    return extracted_text, confidences

def ocr_with_tesseract(img):
    resized_img = image_resize(img)
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image).split("\n")
    confidences = [1.0] * len(extracted_text)  # Tesseract doesn't return confidence scores
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

        # Load previous results and append the new result
        results = load_results()
        new_result = {
            "ocr_method": ocr_method,
            "file_name": file.filename,
            "extracted_text": extracted_text,
            "confidence_scores": confidences
        }
        results.append(new_result)
        save_results(results)

        return new_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server with: uvicorn app:app --host 0.0.0.0 --port 8000
