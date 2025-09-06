# 接口文档

## 函数 `parse_range`

### 描述
解析一个以逗号分隔的数字或范围的字符串，并返回包含这些数字的整数列表。

### 参数
- **s** (Union[str, List]): 要解析的字符串或整数列表。字符串应为逗号分隔的数字，范围用连字符表示。例如：'1,2,5-10'。

### 返回值
- **List[int]**: 一个整数列表，包含解析后的数字。

### 范围说明
- 输入的字符串必须符合特定格式，如单个数字或数字范围。
- 如果输入是一个列表，则直接返回该列表。

---

## 函数 `parse_vec2`

### 描述
解析一个浮点型2维向量，格式为'a,b'。

### 参数
- **s** (Union[str, Tuple[float, float]]): 输入的字符串或元组。字符串格式应为'float,float'。

### 返回值
- **Tuple[float, float]**: 返回解析后的浮点型2维向量。

### 范围说明
- 输入的字符串应包含两个可解析的浮点数。若输入为元组，则直接返回该元组。

---

## 函数 `make_transform`

### 描述
生成一个仿射变换矩阵，用于二维平移和旋转。

### 参数
- **translate** (Tuple[float, float]): 平移的x和y坐标值。
- **angle** (float): 旋转角度，以度为单位。

### 返回值
- **np.ndarray**: 返回一个3x3的numpy数组，表示仿射变换矩阵。

### 范围说明
- `translate`的x和y坐标可以是任意浮点数。
- `angle`可以是任意浮动值，表示旋转的度数。

---

## 函数 `generate_images`

### 描述
使用预训练的网络生成图像。

### 参数
- **network_pkl** (str): 网络的pickle文件名，必须提供。
- **seeds** (List[int]): 随机种子列表，用于确定生成的随机图像，必须提供。
- **truncation_psi** (float, optional): 截断参数，默认值为1。
- **noise_mode** (str, optional): 噪声模式，选择'const', 'random', 'none'中的一种，默认值为'const'。
- **outdir** (str): 输出图像保存的目录，必须提供。
- **translate** (Tuple[float, float], optional): 平移的XY坐标，格式为'0,0'，默认值为(0, 0)。
- **rotate** (float, optional): 旋转角度（以度为单位），默认值为0。
- **class_idx** (Optional[int], optional): 类标签（如果未指定则为无条件生成），默认为None。

### 返回值
- **None**: 此函数不返回任何值。生成的图像将保存在指定的目录中。

### 范围说明
- `network_pkl`必须是一个有效的网络pickle文件URL或文件路径。
- `seeds`应为有效的整数列表，包含预定义的随机种子。
- `truncation_psi`应为一个在0到1之间的浮点数，表示截断值。
- `noise_mode`必须是上述提到的三种模式之一。
- `translate`和`rotate`应为有效的浮点数；`class_idx`应为有效的整数或None（如果使用的是无条件网络）。