以下是根据提供的接口实现信息生成的接口文档：

### 1. `load_cpk`

#### 函数名
`load_cpk`

#### 参数说明
- `checkpoint_path` (str): 检查点文件的路径。
- `model` (torch.nn.Module, optional): 要加载状态字典的模型，默认为 `None`。
- `optimizer` (torch.optim.Optimizer, optional): 要加载状态字典的优化器，默认为 `None`。
- `device` (str): 设备类型（如 "cpu" 或 "cuda"），默认为 "cpu"。

#### 返回值说明
- (int): 返回加载的训练轮次（epoch）。

#### 范围说明
- `checkpoint_path` 应为有效的文件路径。
- `model` 和 `optimizer` 应为有效的 PyTorch 模型和优化器实例。

#### 作用简述
加载给定路径的检查点文件，并将模型和优化器的状态字典加载到相应的对象中。

---

### 2. `Audio2Coeff`

#### 类名
`Audio2Coeff`

#### 初始化信息
- **构造函数**: `__init__(self, sadtalker_path, device)`
  
#### 参数说明
- `sadtalker_path` (dict): 包含模型配置和检查点路径的字典，需包含 `audio2pose_yaml_path`, `audio2exp_yaml_path`, `checkpoint`, `audio2pose_checkpoint`, `audio2exp_checkpoint`, `use_safetensor` 等键。
- `device` (str): 设备类型（如 "cpu" 或 "cuda"）。

#### 属性说明
- `audio2pose_model` (Audio2Pose): 加载的音频到姿态模型。
- `audio2exp_model` (Audio2Exp): 加载的音频到表情模型。
- `device` (str): 当前使用的设备类型。

#### 公有方法
1. **方法名**: `generate`
   - **参数说明**:
     - `batch` (dict): 包含输入数据的字典，需包含 `pic_name` 和 `audio_name`。
     - `coeff_save_dir` (str): 保存系数文件的目录。
     - `pose_style` (int): 指定的姿态风格。
     - `ref_pose_coeff_path` (str, optional): 参考姿态系数文件的路径，默认为 `None`。
   - **返回值说明**:
     - (str): 返回保存的系数文件的完整路径。
   - **范围说明**:
     - `batch` 字典必须包含必要的键。
     - `pose_style` 应为有效的姿态风格 ID。
   - **作用简述**:
     生成音频特征对应的人脸表情系数和姿态系数，并将结果保存为 `.mat` 文件。

2. **方法名**: `using_refpose`
   - **参数说明**:
     - `coeffs_pred_numpy` (np.ndarray): 预测的系数数组。
     - `ref_pose_coeff_path` (str): 参考姿态系数文件的路径。
   - **返回值说明**:
     - (np.ndarray): 更新后的系数数组。
   - **范围说明**:
     - `coeffs_pred_numpy` 应为有效的 NumPy 数组。
   - **作用简述**:
     使用参考姿态系数更新预测的系数数组，以实现相对头部姿态的调整。

---

以上是根据提供的代码实现生成的接口文档，涵盖了函数和类的详细信息。