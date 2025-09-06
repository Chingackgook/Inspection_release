Based on the provided API documentation for the `FaceRestoreHelper` class, here is the classification of the functions and methods:

### Top-Level Functions:
There are no top-level functions explicitly mentioned in the provided documentation. All the functions are methods belonging to the `FaceRestoreHelper` class.

### Methods and Their Class Association:
All methods listed are instance methods of the `FaceRestoreHelper` class. Here is the breakdown:

1. **`__init__`** (instance method)
2. **`set_upscale_factor`** (instance method)
3. **`read_image`** (instance method)
4. **`init_dlib`** (instance method)
5. **`get_face_landmarks_5_dlib`** (instance method)
6. **`get_face_landmarks_5`** (instance method)
7. **`align_warp_face`** (instance method)
8. **`get_inverse_affine`** (instance method)
9. **`add_restored_face`** (instance method)
10. **`paste_faces_to_input_image`** (instance method)
11. **`clean_all`** (instance method)

### Total Number of Interface Classes:
There is **1 interface class** identified in the documentation, which is the `FaceRestoreHelper` class.

Let's address your questions one by one:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
In the `create_interface_objects` method, you need to initialize the `FaceRestoreHelper` class object. Since this is the only class mentioned in the interface documentation, you will create an instance of `FaceRestoreHelper` using the provided `kwargs`. Initialization is necessary because this object will be used to call the instance methods in the `run` method.

### Q2: Which top-level functions should be mapped to `run`?
There are no top-level functions mentioned in the provided interface documentation. Therefore, there are no top-level functions to map to the `run` method.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`?
You should map the following instance methods from the `FaceRestoreHelper` class to the `run` method:

1. `set_upscale_factor(**kwargs)` - This method allows you to set the upscale factor.
2. `read_image(img)` - This method reads an image from a file path or a loaded image.
3. `init_dlib(detection_path, landmark5_path)` - This method initializes the dlib face detection and landmark prediction models.
4. `get_face_landmarks_5_dlib(only_keep_largest=False, scale=1)` - This method detects face landmarks using the dlib model.
5. `get_face_landmarks_5(only_keep_largest=False, only_center_face=False, resize=None, blur_ratio=0.01, eye_dist_threshold=None)` - This method detects face landmarks using the specified detection model.
6. `align_warp_face(save_cropped_path=None, border_mode='constant')` - This method aligns and warps detected faces.
7. `get_inverse_affine(save_inverse_affine_path=None)` - This method calculates and stores the inverse affine matrices.
8. `add_restored_face(restored_face, input_face=None)` - This method adds a restored face image to the list of restored faces.
9. `paste_faces_to_input_image(save_path=None, upsample_img=None, draw_box=False, face_upsampler=None)` - This method pastes the restored faces back onto the input image.
10. `clean_all()` - This method clears all stored landmarks, restored faces, affine matrices, cropped faces, and detected faces.

These methods should be mapped in the `run` method using the format `run(method_name, **kwargs)` since they are all instance methods of the `FaceRestoreHelper` class.