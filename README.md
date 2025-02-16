# TASK 1 - OCR API

This is a FastAPI-based OCR (Optical Character Recognition) API that allows users to extract text from images using multiple OCR libraries, including PaddleOCR, EasyOCR, KerasOCR, and TesseractOCR. The extracted text is returned in JSON format and is also saved locally for reference.

## Features

- Supports multiple OCR engines: PaddleOCR, EasyOCR, KerasOCR, and TesseractOCR
- Accepts image uploads in various formats (JPG, PNG, WEBP, etc.)
- Returns extracted text along with confidence scores
- Saves extracted text in JSON format on the local directory

## Installation

### Prerequisites

Make sure you have Python installed (preferably Python 3.8 or later).

### Clone the Repository

```bash
git clone https://github.com/yourusername/ocr-llm-api.git
cd ocr-llm-api
```

### Install Dependencies

```bash
pip install -r requirements.txt
```


## Running the API

To start the FastAPI server, run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000).

## API Endpoint

### Extract Text from Image

**Endpoint:** `POST /extract-text/`

#### Request Parameters:

- `ocr_method` (query parameter) - Choose one of: `PaddleOCR`, `EasyOCR`, `KerasOCR`, `TesseractOCR`
- `file` (multipart form-data) - The image file to be processed

#### Example Request using cURL:

```bash
curl -X 'POST' 
  'http://localhost:8000/extract-text/?ocr_method=PaddleOCR' 
  -H 'accept: application/json' 
  -H 'Content-Type: multipart/form-data' 
  -F 'file=@sample.webp;type=image/webp'
```

#### Example Response:

```json
{
  "ocr_method": "PaddleOCR",
  "file_name": "sample.webp",
  "extracted_text": [
    "Text line 1",
    "Text line 2",
    "Text line 3"
  ],
  "confidence_scores": [
    0.97,
    0.89,
    0.85
  ]
}
```

## Saving Extracted Text

The extracted text will be automatically saved in a JSON file under the `output/` directory.

Each new request will append or update the extracted text based on the filename.

**Example saved file path:** `output/ocr_results.json`

### Example Saved JSON File:

```json
{
  "ocr_method": "PaddleOCR",
  "file_name": "sample.webp",
  "extracted_text": [
    "Text line 1",
    "Text line 2",
    "Text line 3"
  ],
  "confidence_scores": [
    0.97,
    0.89,
    0.85
  ]
}
```

## Swagger UI and API Documentation

Once the server is running, you can view the interactive API documentation at:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)


---

# TASK 2 - PROPOSE LLM

I have chosen **DistilBERT** as the foundational LLM for text classification due to its efficiency, lightweight architecture, and high performance in natural language processing (NLP) tasks. DistilBERT is a distilled version of BERT that retains 97% of BERT’s performance while being 60% faster and requiring significantly fewer computational resources. This makes it ideal for classifying extracted text as spam or not spam in real-time OCR applications.
[reference](https://arxiv.org/pdf/1910.01108)


## Steps for Fine-Tuning or Prompt Engineering

### Data Preparation:
- Gather a dataset of spam and non-spam text samples.
- Preprocess the text (cleaning, tokenization, and padding).
- Split data into training (80%) and validation (20%) sets.

### Fine-Tuning DistilBERT:
1. Load the pre-trained DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`).
2. Apply transfer learning by training the model on the spam dataset.
3. Use a classification head (fully connected layer) on top of DistilBERT for binary classification.
4. Implement cross-entropy loss and optimize with AdamW.
5. Train the model for **2 epochs** with a **learning rate of 2e-5**.
6. Evaluate performance using precision, recall, F1-score, and accuracy.
7. Achieve an accuracy of **99.1%** on the test set.

### Performance Metrics:
- **Precision:** 99%
- **Recall:** 94%
- **F1-score:** 97%
- **Test Accuracy:** 99.1%
- **Confusion Matrix Analysis:** High accuracy in detecting spam with minimal false positives.
- **AUC-ROC Score:** 0.99
