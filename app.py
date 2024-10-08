import os
import uvicorn
import cv2
import numpy as np
import io
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
import matplotlib.pyplot as plt
from pydantic import BaseModel
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter



app = FastAPI()

# Function to apply unsharp masking to an image
def apply_unsharp_masking(image: Image, masking_level: str):
    if masking_level == "low":
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    elif masking_level == "medium":
        return image.filter(ImageFilter.UnsharpMask(radius=5, percent=200, threshold=5))
    elif masking_level == "high":
        return image.filter(ImageFilter.UnsharpMask(radius=8, percent=230, threshold=10))
    return image

@app.post("/upload/unsharp_masking/")
async def unsharp_masking(file: UploadFile = File(...), masking_level: str = "medium"):
    # Load the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Apply the unsharp masking
    processed_image = apply_unsharp_masking(image, masking_level)
    
    # Save the processed image to an in-memory file
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Return the image as a StreamingResponse
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

# Function for image normalization
def normalize_image(image: np.ndarray) -> io.BytesIO:
    # Normalize the image to the range [0, 1]
    normalized_image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

    # Convert normalized image to uint8 format (0-255)
    normalized_image = (normalized_image * 255).astype(np.uint8)

    is_success, buffer = cv2.imencode(".png", normalized_image)
    io_buf = io.BytesIO(buffer)
    return io_buf

@app.post("/upload/normalize/")
async def upload_image_normalize(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Unable to decode image"}

    result_img = normalize_image(img)
    return StreamingResponse(result_img, media_type="image/png")


# Function for color space conversion
def color_space_conversion(image: np.ndarray, color_space: str) -> io.BytesIO:
    if color_space == 'gray':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'hsv':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'lab':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif color_space == 'yuv':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    else:
        return None  # Invalid color space

    is_success, buffer = cv2.imencode(".png", converted_image)
    io_buf = io.BytesIO(buffer)
    return io_buf

@app.post("/upload/color-space/")
async def upload_image_color_space(file: UploadFile = File(...), color_space: str = 'gray'):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Unable to decode image"}

    result_img = color_space_conversion(img, color_space)

    if result_img is None:
        return {"error": "Invalid color space"}

    return StreamingResponse(result_img, media_type="image/png")


# Function for contour-based segmentation
def contour_segmentation(image: np.ndarray) -> io.BytesIO:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)  # Draw contours in green

    # Combine the original image with the mask
    segmented_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    is_success, buffer = cv2.imencode(".png", segmented_image)
    io_buf = io.BytesIO(buffer)
    return io_buf

@app.post("/upload/contour/")
async def upload_image_contour(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Unable to decode image"}

    result_img = contour_segmentation(img)
    return StreamingResponse(result_img, media_type="image/png")



# Initialize the Haar Cascade for face detection
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class Faces(BaseModel):
    faces: List[Tuple[int, int, int, int]]
    count: int

# Function to equalize histogram
def equalize_image(image, mode='grayscale'):
    if mode == 'grayscale':
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_equalized = cv2.equalizeHist(img_gray)
        original_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        equalized_hist = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
    elif mode == 'rgb':
        img_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0])
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        equalized_gray = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2GRAY)
        equalized_hist = cv2.calcHist([equalized_gray], [0], None, [256], [0, 256])

    is_success, buffer = cv2.imencode(".png", img_equalized)
    io_buf = io.BytesIO(buffer)
    return io_buf, original_hist, equalized_hist

# Function to plot histogram
def plot_histogram(hist):
    fig = plt.figure()
    plt.plot(hist)
    plt.xlim([0, 256])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Function to blur only detected faces
def blur_faces(image, blur_level):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        gray, 
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    kernel_size = (5, 5)
    if blur_level == 'medium':
        kernel_size = (15, 15)
    elif blur_level == 'high':
        kernel_size = (25, 25)

    # Blur only the detected faces
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_blurred = cv2.GaussianBlur(face_roi, kernel_size, 0)
        image[y:y+h, x:x+w] = face_blurred

    is_success, buffer = cv2.imencode(".png", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

# Function for edge detection
def edge_detection(image, method='roberts'):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'roberts':
        kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
        edge_x = cv2.filter2D(img_gray, -1, kernel_x)
        edge_y = cv2.filter2D(img_gray, -1, kernel_y)
        result = edge_x + edge_y

    elif method == 'sobel':
        result = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)

    elif method == 'prewitt':
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edge_x = cv2.filter2D(img_gray, -1, kernel_x)
        edge_y = cv2.filter2D(img_gray, -1, kernel_y)
        result = edge_x + edge_y

    elif method == 'canny':
        result = cv2.Canny(img_gray, 100, 200)

    elif method == 'log':
        result = cv2.Laplacian(img_gray, cv2.CV_64F)

    is_success, buffer = cv2.imencode(".png", result)
    io_buf = io.BytesIO(buffer)
    return io_buf

# Function for face detection
def detect_faces(image: np.ndarray, max_faces: int = 10) -> Tuple[io.BytesIO, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        gray, 
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_count = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{face_count} Face(s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if face_count == 0:
        cv2.putText(image, "No Faces Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    is_success, buffer = cv2.imencode(".png", image)
    io_buf = io.BytesIO(buffer)

    return io_buf, face_count

@app.get("/", response_class=HTMLResponse)
async def render_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/style.css", response_class=HTMLResponse)
async def get_css():
    with open("style.css", "r") as f:
        return HTMLResponse(content=f.read(), media_type="text/css")

@app.post("/upload/equalize/")
async def upload_image_equalize(file: UploadFile = File(...), mode: str = 'grayscale', type: str = None):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if type == 'original_hist':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        original_hist_img = plot_histogram(original_hist)
        return StreamingResponse(original_hist_img, media_type="image/png")
    elif type == 'equalized_hist':
        _, _, equalized_hist = equalize_image(img, mode)
        equalized_hist_img = plot_histogram(equalized_hist)
        return StreamingResponse(equalized_hist_img, media_type="image/png")

    result_img, _, _ = equalize_image(img, mode)
    return StreamingResponse(result_img, media_type="image/png")

@app.post("/upload/blur/")
async def upload_image_blur(file: UploadFile = File(...), blur_level: str = 'low'):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result_img = blur_faces(img, blur_level)

    return StreamingResponse(result_img, media_type="image/png")

@app.post("/upload/edge/")
async def upload_image_edge(file: UploadFile = File(...), method: str = 'roberts'):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result_img = edge_detection(img, method)

    return StreamingResponse(result_img, media_type="image/png")

@app.post("/upload_face/")
async def upload_image_face(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Unable to decode image"}

    result_img, face_count = detect_faces(img)
    headers = {"X-Face-Count": str(face_count)}

    return StreamingResponse(result_img, media_type="image/png", headers=headers)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
