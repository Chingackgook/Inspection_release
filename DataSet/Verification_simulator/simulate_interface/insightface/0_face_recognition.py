import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.insightface import ENV_DIR
from Inspection.adapters.custom_adapters.insightface import *

exe = Executor('insightface', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import cv2
import insightface
import numpy as np

# Initialize face analysis model
exe.create_interface_objects(providers=['CPUExecutionProvider'])  # Load the model

# Prepare the model
exe.run("prepare", ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run the get function
    faces = exe.run("get", img=img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65):  # Adjust this threshold according to your use case.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

def draw_faces_on_image(image_path, faces):
    """Draw detected faces on the input image and return the result"""
    img = cv2.imread(image_path)
    rimg = exe.run("draw_on", img=img, faces=faces)
    return rimg

# Paths to your Indian face images
image1_path = os.path.join(ENV_DIR, "face1.jpg")
image2_path = os.path.join(ENV_DIR, "face2.jpg")

try:
    # Get embeddings
    emb1 = get_face_embedding(image1_path)
    emb2 = get_face_embedding(image2_path)
    
    # Compare faces
    similarity_score, is_same_person = compare_faces(emb1, emb2)
    
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Same person? {'YES' if is_same_person else 'NO'}")
    
    # Optionally draw faces on the first image
    faces = exe.run("get", img=cv2.imread(image1_path))
    rimg = draw_faces_on_image(image1_path, faces)
    cv2.imwrite(os.path.join(FILE_RECORD_PATH, "face1_output.jpg"), rimg)

except Exception as e:
    print(f"Error: {str(e)}")
