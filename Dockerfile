# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for Tesseract and OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx tesseract-ocr && \
    apt-get clean

# Install any needed Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn numpy opencv-python pillow easyocr keras-ocr paddlepaddle pytesseract

# Expose the port the app runs on
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
