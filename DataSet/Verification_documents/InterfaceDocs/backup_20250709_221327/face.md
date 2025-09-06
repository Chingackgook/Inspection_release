# 接口文档

## 类：face_recognition

### 方法：face_distance
- **函数名**: `face_distance`
- **参数说明**:
  - `face_encodings`: List of face encodings to compare (numpy.ndarray).
  - `face_to_compare`: A face encoding to compare against (numpy.ndarray).
- **返回值说明**: A numpy ndarray with the distance for each face in the same order as the 'faces' array (numpy.ndarray).
- **范围说明**: 计算给定面部编码与要比较的面部编码之间的欧几里得距离。

### 方法：load_image_file
- **函数名**: `load_image_file`
- **参数说明**:
  - `file`: Image file name or file object to load (str or file object).
  - `mode`: Format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported (str).
- **返回值说明**: Image contents as numpy array (numpy.ndarray).
- **范围说明**: 加载图像文件并将其转换为numpy数组。

### 方法：face_locations
- **函数名**: `face_locations`
- **参数说明**:
  - `img`: An image (as a numpy array) (numpy.ndarray).
  - `number_of_times_to_upsample`: How many times to upsample the image looking for faces (int).
  - `model`: Which face detection model to use. "hog" or "cnn" (str).
- **返回值说明**: A list of tuples of found face locations in css (top, right, bottom, left) order (list of tuples).
- **范围说明**: 返回图像中人脸的边界框。

### 方法：face_landmarks
- **函数名**: `face_landmarks`
- **参数说明**:
  - `face_image`: Image to search (numpy.ndarray).
  - `face_locations`: Optionally provide a list of face locations to check (list of tuples).
  - `model`: Optional - which model to use. "large" or "small" (str).
- **返回值说明**: A list of dicts of face feature locations (eyes, nose, etc) (list of dicts).
- **范围说明**: 返回图像中每个面部特征的位置。

### 方法：face_encodings
- **函数名**: `face_encodings`
- **参数说明**:
  - `face_image`: The image that contains one or more faces (numpy.ndarray).
  - `known_face_locations`: Optional - the bounding boxes of each face if you already know them (list of tuples).
  - `num_jitters`: How many times to re-sample the face when calculating encoding (int).
  - `model`: Optional - which model to use. "large" or "small" (str).
- **返回值说明**: A list of 128-dimensional face encodings (one for each face in the image) (list of numpy.ndarray).
- **范围说明**: 返回图像中每个面部的128维编码。

### 方法：compare_faces
- **函数名**: `compare_faces`
- **参数说明**:
  - `known_face_encodings`: A list of known face encodings (list of numpy.ndarray).
  - `face_encoding_to_check`: A single face encoding to compare against the list (numpy.ndarray).
  - `tolerance`: How much distance between faces to consider it a match (float).
- **返回值说明**: A list of True/False values indicating which known_face_encodings match the face encoding to check (list of bool).
- **范围说明**: 比较已知面部编码与候选编码以查看它们是否匹配。

## 调用示例

```python
import face_recognition

# 加载图像文件
image = face_recognition.load_image_file("YangDingwen.jpg")

# 获取人脸位置
face_locations = face_recognition.face_locations(image)

# 获取人脸编码
face_encodings = face_recognition.face_encodings(image, face_locations)

# 比较人脸
results = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
```