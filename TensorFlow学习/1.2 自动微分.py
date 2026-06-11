# TensorFlow自动微分与梯度计算学习
# 主要内容：GradientTape、自动微分机制、自定义训练循环

# 导入TensorFlow库
import tensorflow as tf
import numpy as np

# 自动微分基础
print("=== 自动微分基础 ===")

# 创建变量（可训练）
x = tf.Variable([1.0, 2.0, 3.0])
print(f"变量x: {x}")

# 定义函数
@tf.function  # 使用tf.function加速计算图
def compute_y():
    # y = x^2 + 2x + 1
    return x ** 2 + 2 * x + 1

y = compute_y()
print(f"计算结果y: {y}")

# 使用GradientTape计算梯度
print("\n=== 使用GradientTape计算梯度 ===")

# 创建一个persistent=True的GradientTape用于多次计算梯度
with tf.GradientTape(persistent=True) as tape:
    # 定义计算过程
    y = x ** 2 + 2 * x + 1

# 计算dy/dx
grad = tape.gradient(y, x)
print(f"梯度dy/dx: {grad}")

# 手动验证：d(x^2 + 2x + 1)/dx = 2x + 2
print(f"理论梯度(2x + 2): {2 * x + 2}")

# 删除tape释放内存
del tape

# 监视非变量张量
print("\n=== 监视非变量张量 ===")

# 创建普通张量（默认不监视）
w = tf.constant([1.0, 2.0, 3.0])

with tf.GradientTape() as tape:
    # 不会计算梯度，因为w不是变量
    y = w ** 2

# 梯度为None
grad = tape.gradient(y, w)
print(f"监视普通张量的梯度: {grad}")

# 主动监视
with tf.GradientTape() as tape:
    tape.watch(w)  # 主动监视w
    y = w ** 2

grad = tape.gradient(y, w)
print(f"主动监视后的梯度: {grad}")

# 高阶导数
print("\n=== 高阶导数 ===")

x = tf.Variable(2.0)

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = x ** 2  # y = x^2
    # 一阶导数：dy/dx = 2x
    dy_dx = tape2.gradient(y, x)

# 二阶导数：d²y/dx² = 2
d2y_dx2 = tape1.gradient(dy_dx, x)

print(f"一阶导数 dy/dx = {dy_dx}")
print(f"二阶导数 d²y/dx² = {d2y_dx2}")

# 多元函数梯度
print("\n=== 多元函数梯度 ===")

# 定义两个变量
w = tf.Variable([1.0, 2.0, 3.0])
b = tf.Variable(1.0)

# 定义线性函数 y = wx + b
x = tf.constant([2.0, 3.0, 4.0])

with tf.GradientTape() as tape:
    y = tf.reduce_sum(w * x) + b
    loss = y ** 2  # loss = y^2

# 计算梯度
grad_w, grad_b = tape.gradient(loss, [w, b])

print(f"loss = {loss}")
print(f"关于w的梯度: {grad_w}")
print(f"关于b的梯度: {grad_b}")

# 链式法则
print("\n=== 链式法则 ===")

x = tf.Variable(1.0)

with tf.GradientTape() as tape:
    # 链式：z = y^2, y = x^2, 所以 z = (x^2)^2 = x^4
    y = x ** 2
    z = y ** 2

# dz/dx = dz/dy * dy/dx = 2y * 2x = 4x^3 = 4
grad = tape.gradient(z, x)
print(f"z = y^2 = x^4 = {z.numpy()}")
print(f"dz/dx = {grad}")

# 自定义训练循环
print("\n=== 自定义训练循环 ===")

# 准备数据：y = 2x + 1
X = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_true = tf.constant([[3.0], [5.0], [7.0], [9.0], [11.0]])

# 定义模型
class LinearModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.normal([1, 1]))
        self.b = tf.Variable(tf.zeros([1, 1]))

    def __call__(self, x):
        return x @ self.w + self.b

model = LinearModel()

# 定义优化器
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 训练参数
epochs = 100

# 训练循环
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # 前向传播
        y_pred = model(X)
        # 计算损失（MSE）
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # 反向传播
    gradients = tape.gradient(loss, model.trainable_variables)

    # 更新参数
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 每10轮打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: 损失 = {loss.numpy():.4f}")

# 训练后的结果
print(f"\n训练后的参数:")
print(f"w = {model.w.numpy()[0, 0]:.4f}")
print(f"b = {model.b.numpy()[0, 0]:.4f}")
print(f"理论值: w = 2.0, b = 1.0")

# 测试模型
test_x = tf.constant([[6.0]])
print(f"\n预测 x=6: y = {model(test_x).numpy()[0, 0]:.4f}")
print(f"理论值: y = 13.0")

# 使用GradientTape和Keras层
print("\n=== 与Keras层结合 ===")

# 使用Keras的Dense层
layer = tf.keras.layers.Dense(1, input_shape=(1,))
layer.build(input_shape=(None, 1))

# 准备数据
X = tf.constant([[1.0], [2.0], [3.0]])
y_true = tf.constant([[2.0], [4.0], [6.0]])

# 优化器
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# 训练循环
for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = layer(X)
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    gradients = tape.gradient(loss, layer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, layer.trainable_variables))

print(f"\nKeras层训练后的权重: {layer.kernel.numpy()[0, 0]:.4f}")
print(f"理论值: 2.0")

# 停止梯度
print("\n=== 停止梯度 ===")

x = tf.Variable([1.0, 2.0, 3.0])

with tf.GradientTape() as tape:
    y = x ** 2
    # 使用tf.stop_gradient阻止梯度传递
    y_stopped = tf.stop_gradient(y)

# y有梯度，y_stopped没有
print(f"y的梯度: {tape.gradient(y, x)}")
print(f"y_stopped的梯度: {tape.gradient(y_stopped, x)}")