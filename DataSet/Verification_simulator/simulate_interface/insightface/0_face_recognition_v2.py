from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.insightface import *
exe = Executor('insightface', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# Import the existing package
import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
# end

exe.create_interface_objects(interface_class_name='FaceAnalysis', name='buffalo_l', providers=['CPUExecutionProvider'])
_ = exe.run('prepare', ctx_id=-1)

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')
    faces = exe.run('get', img=img)
    if len(faces) < 1:
        raise ValueError('No faces detected in the image')
    if len(faces) > 1:
        print('Warning: Multiple faces detected. Using first detected face')
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65):
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return (similarity, similarity > threshold)

# Main logic
image1_path = RESOURCES_PATH + 'images/test_image.jpg'# Parts that may need manual modification:
image2_path = RESOURCES_PATH + 'images/test_image.jpg'# Parts that may need manual modification:
# end

emb1 = get_face_embedding(image1_path)
emb2 = get_face_embedding(image2_path)
similarity_score, is_same_person = compare_faces(emb1, emb2)
print(f'Similarity Score: {similarity_score:.4f}')
print(f"Same person? {('YES' if is_same_person else 'NO')}")