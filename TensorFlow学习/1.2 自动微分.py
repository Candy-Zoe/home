# TensorFlow自动微分学习
# 主要内容：tf.GradientTape、梯度计算

import tensorflow as tf

print("=== 基本梯度计算 ===")
x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 3 * x + 1

dy_dx = tape.gradient(y, x)
print(f"x = {x.numpy()}")
print(f"y = x^2 + 3x + 1 = {y.numpy()}")
print(f"dy/dx = {dy_dx.numpy()}")

print("\n=== 多变量梯度 ===")
x = tf.Variable(1.0)
y = tf.Variable(2.0)

with tf.GradientTape() as tape:
    z = x * y + x ** 2

dz_dx, dz_dy = tape.gradient(z, [x, y])
print(f"z = x*y + x^2 = {z.numpy()}")
print(f"dz/dx = {dz_dx.numpy()}")
print(f"dz/dy = {dz_dy.numpy()}")

print("\n=== 高阶导数 ===")
x = tf.Variable(1.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)

print(f"y = x^3")
print(f"dy/dx = {dy_dx.numpy()}")
print(f"d2y/dx2 = {d2y_dx2.numpy()}")

print("\n=== 持久化tape ===")
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = y ** 2

dy_dx = tape.gradient(y, x)
dz_dx = tape.gradient(z, x)

print(f"dy/dx = {dy_dx.numpy()}")
print(f"dz/dx = {dz_dx.numpy()}")