# TensorFlow自定义层与模型学习
# 主要内容：自定义层、自定义模型、混合精度训练

import tensorflow as tf
from tensorflow.keras import layers, models

print("=== 自定义层 ===")
class CustomDense(layers.Layer):
    def __init__(self, units=32, activation=None):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

print("自定义层已创建")

print("\n=== 自定义模型 ===")
class CustomModel(models.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = CustomDense(64, activation='relu')
        self.dense2 = layers.Dense(10)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = CustomModel()
print(f"模型摘要:\n{model.summary()}")

print("\n=== 测试模型 ===")
input_data = tf.random.normal((1, 28, 28, 1))
output = model(input_data)
print(f"输出形状: {output.shape}")

print("\n=== 混合精度训练 ===")
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model = CustomModel()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print("混合精度模型已编译")

print("\n=== 自定义训练循环 ===")
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

print("自定义训练步骤已定义")

print("\n=== 保存自定义模型 ===")
model.save('custom_model.h5')
print("自定义模型已保存")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('custom_model.h5'):
    os.remove('custom_model.h5')
    print("已删除测试模型")