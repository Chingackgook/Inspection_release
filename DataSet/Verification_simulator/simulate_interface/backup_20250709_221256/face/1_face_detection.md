为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式，并确定如何构造相应的参数。以下是对每个关键函数的分析和替换方案：

### 1. 函数分析与替换

#### 1.1 `load_image_file`
- **原调用**: `unknown_image = face_recognition.load_image_file(image_to_check)`
- **替换形式**: `unknown_image = exe.run("load_image_file", file=image_to_check)`
- **参数**: `file` 是图像文件的路径。

#### 1.2 `face_locations`
- **原调用**: `face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=upsample, model=model)`
- **替换形式**: `face_locations = exe.run("face_locations", img=unknown_image, number_of_times_to_upsample=upsample, model=model)`
- **参数**: `img` 是加载的图像，`number_of_times_to_upsample` 和 `model` 是从命令行参数获取的。

#### 1.3 `face_encodings`
- **原调用**: `face_encodings = face_recognition.face_encodings(unknown_image, face_locations)`
- **替换形式**: `face_encodings = exe.run("face_encodings", face_image=unknown_image, known_face_locations=face_locations)`
- **参数**: `face_image` 是加载的图像，`known_face_locations` 是检测到的人脸位置。

#### 1.4 `compare_faces`
- **原调用**: `results = face_recognition.compare_faces(known_face_encodings, face_encodings[0])`
- **替换形式**: `results = exe.run("compare_faces", known_face_encodings=known_face_encodings, face_encoding_to_check=face_encodings[0])`
- **参数**: `known_face_encodings` 是已知的人脸编码，`face_encoding_to_check` 是要比较的编码。

#### 1.5 `face_landmarks`
- **原调用**: 该函数在示例代码中未直接调用，但可以在需要时使用。
- **替换形式**: `landmarks = exe.run("face_landmarks", face_image=unknown_image, face_locations=face_locations)`
- **参数**: `face_image` 是加载的图像，`face_locations` 是检测到的人脸位置。

#### 1.6 `face_distance`
- **原调用**: 该函数在示例代码中未直接调用，但可以在需要时使用。
- **替换形式**: `distances = exe.run("face_distance", face_encodings=known_face_encodings, face_to_compare=face_encodings[0])`
- **参数**: `face_encodings` 是已知的人脸编码，`face_to_compare` 是要比较的编码。

### 2. 模拟输入方案

为了测试和验证替换后的代码，我们需要提供模拟输入。以下是一个方案：

- **输入图像**: 提供一张名为 `test_image.jpg` 的图像文件，作为 `image_to_check` 参数。
- **已知人脸编码**: 创建一个模拟的已知人脸编码列表 `known_face_encodings`，可以是随机生成的128维数组。
- **命令行参数**:
  - `--cpus`: 设置为 `1`，表示使用单核处理。
  - `--model`: 设置为 `"hog"`，使用HOG模型进行人脸检测。
  - `--upsample`: 设置为 `1`，表示对图像进行一次上采样以检测人脸。

### 3. 总结

通过上述分析，我们可以将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，并为其提供模拟输入以进行测试。这样可以确保代码在新的执行环境中正常运行，并且能够正确调用封装的函数。