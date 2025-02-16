# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies for OCR and image processing
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 tesseract-ocr && \
    apt-get clean

# Set UTF-8 encoding for text processing stability
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
