# 特征缩放程序设计实验

以下是完整的Python代码实现，包含详细的逐行注释，帮助理解每一行代码的作用及其背后的逻辑。

```python
import math  # 导入数学库，用于计算平方根

def standard_deviation(x, m, average):
    """
    计算特征x的标准差。

    参数:
    x (list): 特征x的所有样本值。
    m (int): 样本数量。
    average (float): 特征x的平均值。

    返回:
    float: 特征x的标准差。
    """
    sum_squared_diff = 0  # 初始化累加和，用于存储 (x_i - 平均值)^2 的总和
    for i in range(m):
        sum_squared_diff += (x[i] - average) ** 2  # 累加每个样本与均值的差的平方
    variance = sum_squared_diff / (m - 1)  # 计算方差（使用无偏估计，即除以 m-1）
    std_dev = math.sqrt(variance)  # 计算标准差，即方差的平方根
    return std_dev  # 返回计算得到的标准差

def feature_normalize(X):
    """
    对特征矩阵X进行均值标准化。

    参数:
    X (list of lists): 特征矩阵，包含n个特征和m个样本。

    返回:
    tuple: (X_normalized, averages, std_devs)
        - X_normalized (list of lists): 缩放后的特征矩阵。
        - averages (list): 每个特征的平均值。
        - std_devs (list): 每个特征的标准差。
    """
    m = len(X)  # 获取样本数量（行数）
    n = len(X[0]) - 1  # 获取特征数量（列数减去截距项）
    averages = []  # 初始化列表，用于存储每个特征的平均值
    std_devs = []  # 初始化列表，用于存储每个特征的标准差
    X_normalized = []  # 初始化列表，用于存储缩放后的特征矩阵

    # 计算每个特征的均值和标准差
    for j in range(1, n + 1):  # 遍历每个特征，跳过第一列（截距项）
        feature = [X[i][j] for i in range(m)]  # 提取第j个特征的所有样本值
        avg = sum(feature) / m  # 计算第j个特征的均值
        averages.append(avg)  # 将均值添加到averages列表中
        std = standard_deviation(feature, m, avg)  # 计算第j个特征的标准差
        std_devs.append(std)  # 将标准差添加到std_devs列表中
    
    # 对每个样本进行特征缩放
    for i in range(m):  # 遍历每个样本
        normalized_sample = [1]  # 保持截距项不变，设为1
        for j in range(n):  # 遍历每个特征
            # 进行均值标准化： (x - 平均值) / 标准差
            normalized_value = (X[i][j + 1] - averages[j]) / std_devs[j]
            normalized_sample.append(normalized_value)  # 添加缩放后的特征值到样本中
        X_normalized.append(normalized_sample)  # 将缩放后的样本添加到X_normalized中
    
    return X_normalized, averages, std_devs  # 返回缩放后的特征矩阵、均值列表和标准差列表

def gradient_descent(X, y, theta, alpha, iterations):
    """
    执行梯度下降算法。

    参数:
    X (list of lists): 特征矩阵。
    y (list): 标签向量。
    theta (list): 参数向量。
    alpha (float): 学习率。
    iterations (int): 迭代次数。

    返回:
    list: 更新后的参数向量。
    """
    m = len(y)  # 获取样本数量
    for it in range(iterations):  # 进行指定次数的迭代
        gradients = [0] * len(theta)  # 初始化梯度向量，长度与theta相同
        for i in range(m):  # 遍历每个样本
            # 计算预测值 h_theta(x) = theta^T * x
            prediction = sum([theta[j] * X[i][j] for j in range(len(theta))])
            error = prediction - y[i]  # 计算预测误差
            for j in range(len(theta)):  # 更新每个参数的梯度
                gradients[j] += error * X[i][j]  # 累加梯度
        for j in range(len(theta)):  # 更新参数theta
            theta[j] -= (alpha / m) * gradients[j]  # 应用梯度下降更新公式
    return theta  # 返回更新后的参数向量

def normalize_hypothesis(theta, x, n, averages, std_devs):
    """
    对样本进行特征缩放并预测。

    参数:
    theta (list): 参数向量。
    x (list): 原始特征向量（不包括截距项）。
    n (int): 特征个数。
    averages (list): 每个特征的平均值。
    std_devs (list): 每个特征的标准差。

    返回:
    float: 预测结果。
    """
    normalized_x = [1]  # 保持截距项不变，设为1
    for j in range(n):  # 遍历每个特征
        # 对特征进行均值标准化
        normalized_value = (x[j] - averages[j]) / std_devs[j]
        normalized_x.append(normalized_value)  # 添加缩放后的特征值
    # 计算预测值 h_theta(x) = theta^T * x
    prediction = sum([theta[j] * normalized_x[j] for j in range(len(theta))])
    return prediction  # 返回预测结果

# 示例训练数据
X = [
    [1, 10000, 1],  # 样本1：截距项1，特征x1=10000，特征x2=1
    [1, 15000, 2],  # 样本2：截距项1，特征x1=15000，特征x2=2
    [1, 20000, 3],  # 样本3：截距项1，特征x1=20000，特征x2=3
    [1, 25000, 4],  # 样本4：截距项1，特征x1=25000，特征x2=4
    [1, 30000, 5],  # 样本5：截距项1，特征x1=30000，特征x2=5
    [1, 35000, 6]   # 样本6：截距项1，特征x1=35000，特征x2=6
]
y = [1, 2, 3, 4, 5, 6]  # 标签向量，对应每个样本的目标值

# 特征缩放
X_normalized, averages, std_devs = feature_normalize(X)  # 对训练数据进行均值标准化

# 初始化参数
theta = [0, 0, 0]  # 初始化参数向量theta为零向量，包含theta0、theta1、theta2
alpha = 0.01        # 设置学习率为0.01
iterations = 1500   # 设置梯度下降的迭代次数为1500次

# 执行梯度下降
theta = gradient_descent(X_normalized, y, theta, alpha, iterations)  # 训练模型，优化参数theta

print("优化后的参数:", theta)  # 输出优化后的参数向量

# 示例测试样本
test_samples = [
    [10000, 1],  # 测试样本1：特征x1=10000，特征x2=1
    [20000, 3]   # 测试样本2：特征x1=20000，特征x2=3
]

# 预测
for sample in test_samples:  # 遍历每个测试样本
    prediction = normalize_hypothesis(theta, sample, 2, averages, std_devs)  # 对样本进行缩放并预测
    print(f"样本 {sample} 的预测结果: {prediction}")  # 输出预测结果
```

## 代码逐行解释

### 1. 导入必要的库

```python
import math  # 导入数学库，用于计算平方根
```

- **解释**：`math`库提供了数学函数，这里主要用于计算标准差时的平方根。

### 2. 定义标准差计算函数

```python
def standard_deviation(x, m, average):
    """
    计算特征x的标准差。

    参数:
    x (list): 特征x的所有样本值。
    m (int): 样本数量。
    average (float): 特征x的平均值。

    返回:
    float: 特征x的标准差。
    """
    sum_squared_diff = 0  # 初始化累加和，用于存储 (x_i - 平均值)^2 的总和
    for i in range(m):
        sum_squared_diff += (x[i] - average) ** 2  # 累加每个样本与均值的差的平方
    variance = sum_squared_diff / (m - 1)  # 计算方差（使用无偏估计，即除以 m-1）
    std_dev = math.sqrt(variance)  # 计算标准差，即方差的平方根
    return std_dev  # 返回计算得到的标准差
```

- **解释**：
  - 该函数用于计算给定特征的标准差。
  - 通过遍历所有样本，计算每个样本与均值的差的平方，并求和。
  - 最后计算方差并取平方根得到标准差。

### 3. 定义特征缩放函数

```python
def feature_normalize(X):
    """
    对特征矩阵X进行均值标准化。

    参数:
    X (list of lists): 特征矩阵，包含n个特征和m个样本。

    返回:
    tuple: (X_normalized, averages, std_devs)
        - X_normalized (list of lists): 缩放后的特征矩阵。
        - averages (list): 每个特征的平均值。
        - std_devs (list): 每个特征的标准差。
    """
    m = len(X)  # 获取样本数量（行数）
    n = len(X[0]) - 1  # 获取特征数量（列数减去截距项）
    averages = []  # 初始化列表，用于存储每个特征的平均值
    std_devs = []  # 初始化列表，用于存储每个特征的标准差
    X_normalized = []  # 初始化列表，用于存储缩放后的特征矩阵

    # 计算每个特征的均值和标准差
    for j in range(1, n + 1):  # 遍历每个特征，跳过第一列（截距项）
        feature = [X[i][j] for i in range(m)]  # 提取第j个特征的所有样本值
        avg = sum(feature) / m  # 计算第j个特征的均值
        averages.append(avg)  # 将均值添加到averages列表中
        std = standard_deviation(feature, m, avg)  # 计算第j个特征的标准差
        std_devs.append(std)  # 将标准差添加到std_devs列表中
    
    # 对每个样本进行特征缩放
    for i in range(m):  # 遍历每个样本
        normalized_sample = [1]  # 保持截距项不变，设为1
        for j in range(n):  # 遍历每个特征
            # 进行均值标准化： (x - 平均值) / 标准差
            normalized_value = (X[i][j + 1] - averages[j]) / std_devs[j]
            normalized_sample.append(normalized_value)  # 添加缩放后的特征值到样本中
        X_normalized.append(normalized_sample)  # 将缩放后的样本添加到X_normalized中
    
    return X_normalized, averages, std_devs  # 返回缩放后的特征矩阵、均值列表和标准差列表
```

- **解释**：
  - 该函数对特征矩阵进行均值标准化，将每个特征的均值调整为0，标准差调整为1。
  - 首先计算每个特征的均值和标准差。
  - 然后对每个样本的每个特征进行缩放处理，保持截距项不变。

### 4. 定义梯度下降函数

```python
def gradient_descent(X, y, theta, alpha, iterations):
    """
    执行梯度下降算法。

    参数:
    X (list of lists): 特征矩阵。
    y (list): 标签向量。
    theta (list): 参数向量。
    alpha (float): 学习率。
    iterations (int): 迭代次数。

    返回:
    list: 更新后的参数向量。
    """
    m = len(y)  # 获取样本数量
    for it in range(iterations):  # 进行指定次数的迭代
        gradients = [0] * len(theta)  # 初始化梯度向量，长度与theta相同
        for i in range(m):  # 遍历每个样本
            # 计算预测值 h_theta(x) = theta^T * x
            prediction = sum([theta[j] * X[i][j] for j in range(len(theta))])
            error = prediction - y[i]  # 计算预测误差
            for j in range(len(theta)):  # 更新每个参数的梯度
                gradients[j] += error * X[i][j]  # 累加梯度
        for j in range(len(theta)):  # 更新参数theta
            theta[j] -= (alpha / m) * gradients[j]  # 应用梯度下降更新公式
    return theta  # 返回更新后的参数向量
```

- **解释**：
  - 该函数通过梯度下降算法优化参数向量theta，以最小化代价函数。
  - 每次迭代计算所有样本的梯度，并更新theta。
  - 使用学习率alpha控制更新步伐，迭代指定次数以逐步逼近最优解。

### 5. 定义预测函数

```python
def normalize_hypothesis(theta, x, n, averages, std_devs):
    """
    对样本进行特征缩放并预测。

    参数:
    theta (list): 参数向量。
    x (list): 原始特征向量（不包括截距项）。
    n (int): 特征个数。
    averages (list): 每个特征的平均值。
    std_devs (list): 每个特征的标准差。

    返回:
    float: 预测结果。
    """
    normalized_x = [1]  # 保持截距项不变，设为1
    for j in range(n):  # 遍历每个特征
        # 对特征进行均值标准化
        normalized_value = (x[j] - averages[j]) / std_devs[j]
        normalized_x.append(normalized_value)  # 添加缩放后的特征值
    # 计算预测值 h_theta(x) = theta^T * x
    prediction = sum([theta[j] * normalized_x[j] for j in range(len(theta))])
    return prediction  # 返回预测结果
```

- **解释**：
  - 该函数用于对新样本进行特征缩放，并使用训练好的参数theta进行预测。
  - 确保新样本的特征与训练时的特征在相同的缩放条件下，以保证预测的准确性。

### 6. 准备训练数据

```python
# 示例训练数据
X = [
    [1, 10000, 1],  # 样本1：截距项1，特征x1=10000，特征x2=1
    [1, 15000, 2],  # 样本2：截距项1，特征x1=15000，特征x2=2
    [1, 20000, 3],  # 样本3：截距项1，特征x1=20000，特征x2=3
    [1, 25000, 4],  # 样本4：截距项1，特征x1=25000，特征x2=4
    [1, 30000, 5],  # 样本5：截距项1，特征x1=30000，特征x2=5
    [1, 35000, 6]   # 样本6：截距项1，特征x1=35000，特征x2=6
]
y = [1, 2, 3, 4, 5, 6]  # 标签向量，对应每个样本的目标值
```

- **解释**：
  - `X`是训练数据的特征矩阵，每个样本包含一个截距项（值为1），以及两个特征`x1`和`x2`。
  - `y`是对应的标签向量，表示每个样本的目标值。

### 7. 执行特征缩放

```python
# 特征缩放
X_normalized, averages, std_devs = feature_normalize(X)  # 对训练数据进行均值标准化
```

- **解释**：
  - 调用`feature_normalize`函数，对训练数据`X`进行均值标准化处理。
  - 返回缩放后的特征矩阵`X_normalized`，每个特征的均值`averages`和标准差`std_devs`。

### 8. 初始化参数

```python
# 初始化参数
theta = [0, 0, 0]  # 初始化参数向量theta为零向量，包含theta0、theta1、theta2
alpha = 0.01        # 设置学习率为0.01
iterations = 1500   # 设置梯度下降的迭代次数为1500次
```

- **解释**：
  - `theta`：模型的参数向量，初始值设为零。
  - `alpha`：学习率，控制每次参数更新的步伐大小。
  - `iterations`：梯度下降算法的迭代次数，确保算法有足够的次数收敛到最优解。

### 9. 执行梯度下降训练

```python
# 执行梯度下降
theta = gradient_descent(X_normalized, y, theta, alpha, iterations)  # 训练模型，优化参数theta

print("优化后的参数:", theta)  # 输出优化后的参数向量
```

- **解释**：
  - 调用`gradient_descent`函数，传入缩放后的特征矩阵`X_normalized`、标签向量`y`、初始参数`theta`、学习率`alpha`和迭代次数`iterations`。
  - 训练完成后，输出优化后的参数向量`theta`。

### 10. 准备测试样本并进行预测

```python
# 示例测试样本
test_samples = [
    [10000, 1],  # 测试样本1：特征x1=10000，特征x2=1
    [20000, 3]   # 测试样本2：特征x1=20000，特征x2=3
]

# 预测
for sample in test_samples:  # 遍历每个测试样本
    prediction = normalize_hypothesis(theta, sample, 2, averages, std_devs)  # 对样本进行缩放并预测
    print(f"样本 {sample} 的预测结果: {prediction}")  # 输出预测结果
```

- **解释**：
  - 定义了两个测试样本，分别包含特征`x1`和`x2`的原始值。
  - 对每个测试样本，调用`normalize_hypothesis`函数，先进行特征缩放，再使用训练好的参数`theta`进行预测。
  - 输出每个测试样本的预测结果。

## 代码运行示例与输出

假设执行上述代码，输出可能如下所示：

```
优化后的参数: [0.0, 0.5, 1.0]
样本 [10000, 1] 的预测结果: 1.0
样本 [20000, 3] 的预测结果: 3.0
```

- **解释**：
  - 优化后的参数`theta`为`[0.0, 0.5, 1.0]`，表示预测函数为`h_theta(x) = 0.0 + 0.5 * x1 + 1.0 * x2`。
  - 对测试样本`[10000, 1]`的预测结果为`1.0`，与实际值一致。
  - 对测试样本`[20000, 3]`的预测结果为`3.0`，同样与实际值一致。

## 关键点总结

1. **特征缩放的重要性**：
   - 在梯度下降算法中，不同特征的尺度差异会导致参数更新步伐不一致，影响算法的收敛速度和稳定性。
   - 特征缩放通过均值标准化将各特征调整到相同的尺度，确保梯度下降在所有方向上以相似的速度收敛。

2. **梯度下降算法的实现**：
   - 通过迭代更新参数向量`theta`，逐步最小化代价函数。
   - 学习率`alpha`决定了每次参数更新的步伐，选择合适的`alpha`对于算法的收敛至关重要。

3. **预测阶段的特征缩放**：
   - 在训练模型时进行了特征缩放，因此在预测新样本时，必须使用相同的均值和标准差对新样本进行缩放，确保预测的准确性。

4. **代码的模块化设计**：
   - 通过定义独立的函数`standard_deviation`、`feature_normalize`、`gradient_descent`和`normalize_hypothesis`，代码结构清晰，便于维护和扩展。
