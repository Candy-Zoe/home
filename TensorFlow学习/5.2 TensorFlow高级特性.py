# TensorFlow 2.x高级特性学习
# 主要内容：自定义训练循环、分布策略、TPU支持、模型集成

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("=== 检查GPU/TPU可用性 ===")
print(f"GPU可用: {tf.config.list_physical_devices('GPU')}")
print(f"TPU可用: {len(tf.config.list_logical_devices('TPU')) > 0}")

print("\n=== 分布策略 ===")
strategy = tf.distribute.MirroredStrategy()
print(f"副本数量: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10)
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print("分布策略模型已创建")

print("\n=== 自定义训练循环 ===")
class CustomTrainingModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(10)
    
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

print("自定义训练模型已定义")

print("\n=== Gradient Clipping ===")
optimizer = keras.optimizers.Adam(clipnorm=1.0)
print("梯度裁剪已配置")

print("\n=== 学习率调度 ===")
lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

schedule_visualization = [lr_scheduler(step).numpy() for step in range(10000)]
plt.plot(schedule_visualization)
plt.title('学习率衰减')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.show()

print("\n=== 模型集成 ===")
def create_model():
    return keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.Dense(10)
    ])

models = [create_model() for _ in range(3)]
print(f"模型数量: {len(models)}")

print("\n=== 模型剪枝 ===")
prune_low_magnitude = keras.optimizers.squash.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=1000,
    end_step=5000
)

print("剪枝调度已创建")

print("\n=== 知识蒸馏 ===")
class DistillationLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        teacher_logit, student_logit = y_pred
        temperature = 3.0
        
        soft_teacher = tf.nn.softmax(teacher_logit / temperature)
        soft_student = tf.nn.log_softmax(student_logit / temperature)
        
        distillation_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(soft_teacher, soft_student)
        ) * (temperature ** 2)
        
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, student_logit)
        
        return 0.5 * hard_loss + 0.5 * distillation_loss

print("知识蒸馏损失已定义")

print("\n=== TensorFlow Lite转换 ===")
model = keras.Sequential([keras.layers.Dense(10)])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print(f"TFLite模型大小: {len(tflite_model) / 1024:.2f} KB")

print("\n=== TensorFlow SavedModel ===")
model.save('saved_model')
loaded_model = keras.models.load_model('saved_model')
print("SavedModel已保存和加载")

print("\n=== 清理 ===")
import shutil
import os
if os.path.exists('saved_model'):
    shutil.rmtree('saved_model')
    print("已删除SavedModel")