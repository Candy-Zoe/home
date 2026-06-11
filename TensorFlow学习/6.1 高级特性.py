# TensorFlow高级特性学习
# 主要内容：自定义层、自定义训练循环、分布式训练、模型部署、TensorFlow Lite

# 导入必要的库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow基础
print("=== TensorFlow基础 ===")
print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU可用: {tf.config.list_physical_devices('GPU')}")

# 张量操作
print("\n=== 张量操作 ===")

# 创建张量
scalar = tf.constant(42)
vector = tf.constant([1, 2, 3, 4, 5])
matrix = tf.constant([[1, 2], [3, 4]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"标量: {scalar}, 形状: {scalar.shape}")
print(f"向量: {vector.numpy()}, 形状: {vector.shape}")
print(f"矩阵:\n{matrix.numpy()}")
print(f"3D张量形状: {tensor_3d.shape}")

# 特殊张量
print("\n特殊张量创建:")
zeros = tf.zeros([3, 3])
ones = tf.ones([2, 4])
eye = tf.eye(3)
fill = tf.fill([2, 3], 7)
random_uniform = tf.random.uniform([2, 3], 0, 1)
random_normal = tf.random.normal([2, 3], 0, 1)

print(f"全零张量:\n{zeros.numpy()}")
print(f"全一张量:\n{ones.numpy()}")
print(f"单位矩阵:\n{eye.numpy()}")
print(f"填充张量:\n{fill.numpy()}")

# 张量运算
print("\n张量运算:")
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print(f"矩阵加法:\n{tf.add(a, b).numpy()}")
print(f"矩阵乘法:\n{tf.matmul(a, b).numpy()}")
print(f"元素乘法:\n{tf.multiply(a, b).numpy()}")
print(f"矩阵转置:\n{tf.transpose(a).numpy()}")

# 变量和自动微分
print("\n=== 变量和自动微分 ===")

# 创建变量
weight = tf.Variable(tf.random.normal([3, 3]))
bias = tf.Variable(tf.zeros([3]))

print(f"权重变量形状: {weight.shape}")
print(f"偏置变量形状: {bias.shape}")

# 自动微分
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    y = x ** 2
    z = 2 * y + 3

# 计算梯度
gradient = tape.gradient(z, x)
print(f"z = 2x^2 + 3 在 x=3 处的梯度: {gradient.numpy()}")

# 多变量梯度
x = tf.Variable(3.0)
y = tf.Variable(2.0)
with tf.GradientTape() as tape:
    f = x**2 + 2*x*y + y**2

grads = tape.gradient(f, [x, y])
print(f"f = x^2 + 2xy + y^2 对x的梯度: {grads[0].numpy()}")
print(f"f = x^2 + 2xy + y^2 对y的梯度: {grads[1].numpy()}")

# 自定义层
print("\n=== 自定义层 ===")

class DenseLayer(tf.keras.layers.Layer):
    """自定义全连接层"""
    
    def __init__(self, units, activation=None, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # 创建权重
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='kernel'
        )
        # 创建偏置
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        # 前向传播
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            z = self.activation(z)
        return z

# 使用自定义层
custom_dense = DenseLayer(units=10, activation='relu')
test_input = tf.random.normal([1, 5])
output = custom_dense(test_input)
print(f"自定义层输出形状: {output.shape}")

# 自定义残差块
class ResidualBlock(tf.keras.layers.Layer):
    """残差块"""
    
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, inputs, training=False):
        # 残差连接
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        # 残差相加
        return self.relu(x + residual)

# 自定义模型
print("\n=== 自定义模型 ===")

class ResNetBlock(tf.keras.Model):
    """使用自定义层的ResNet模型"""
    
    def __init__(self, num_classes=10):
        super(ResNetBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(2)
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.gap(x)
        return self.fc(x)

# 使用Sequential API
print("\n使用Sequential API:")
model_seq = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_seq.summary()

# 使用Functional API
print("\n使用Functional API:")
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model_func = tf.keras.Model(inputs, outputs, name='functional_model')
model_func.summary()

# 自定义训练循环
print("\n=== 自定义训练循环 ===")

# 准备数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 简化版：只使用前1000个样本
x_train_small = x_train[:1000]
y_train_small = y_train[:1000]
x_test_small = x_test[:200]
y_test_small = y_test[:200]

# 创建简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# 训练数据batch
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train_small, y_train_small)
).shuffle(1000).batch(batch_size)

# 自定义训练循环
print("开始自定义训练循环...")
epochs = 3
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # 训练
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 前向传播
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        
        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 更新权重
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 更新指标
        train_acc_metric.update_state(y_batch, logits)
        
        if step % 10 == 0:
            print(f"  步骤 {step}: 损失 = {loss.numpy():.4f}, 准确率 = {train_acc_metric.result().numpy():.4f}")
    
    # 重置指标
    train_acc_metric.reset_states()

# 评估
print("\n评估模型:")
test_logits = model(x_test_small, training=False)
test_pred = tf.argmax(test_logits, axis=1)
test_acc = tf.reduce_mean(tf.cast(tf.equal(test_pred, y_test_small), tf.float32))
print(f"测试集准确率: {test_acc.numpy():.4f}")

# 回调函数
print("\n=== 回调函数 ===")

# 自定义回调
class CustomCallback(tf.keras.callbacks.Callback):
    """自定义回调函数"""
    
    def on_train_begin(self, logs=None):
        print("训练开始!")
    
    def on_train_end(self, logs=None):
        print("训练结束!")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            print(f"  Epoch {epoch+1}: 损失 = {logs['loss']:.4f}, 准确率 = {logs['accuracy']:.4f}")

# 使用回调训练
print("\n使用回调函数训练:")
model_cb = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_cb.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练（使用更少数据）
history = model_cb.fit(
    x_train_small, y_train_small,
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    callbacks=[CustomCallback()],
    verbose=0
)

# 训练历史可视化
print("\n训练历史可视化:")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='训练损失')
axes[0].plot(history.history['val_loss'], label='验证损失')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='训练准确率')
axes[1].plot(history.history['val_accuracy'], label='验证准确率')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('准确率曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# TensorFlow数据管道
print("\n=== tf.data API ===")

# 从numpy数组创建数据集
print("1. 从numpy创建数据集:")
dataset = tf.data.Dataset.from_tensor_slices((x_train_small, y_train_small))
print(f"数据集元素数量: {len(list(dataset))}")
print(f"第一个元素: x.shape={list(dataset)[0][0].shape}, y={list(dataset)[0][1].numpy()}")

# 数据集操作
print("\n2. 数据集操作:")
batched_dataset = dataset.batch(32)
shuffled_dataset = dataset.shuffle(buffer_size=100)
prefetch_dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
print(f"批处理后: {len(list(batched_dataset))} 批次")
print(f"预取已设置: 自动优化")

# 性能优化
print("\n3. 性能优化技巧:")
print("  - .cache(): 将数据缓存到内存")
print("  - .shuffle(): 打乱数据")
print("  - .batch(): 批处理")
print("  - .prefetch(): 预取数据")
print("  - .map(): 并行数据处理")

# 模型保存和加载
print("\n=== 模型保存和加载 ===")

# 保存整个模型
model.save('saved_model.h5')
print("模型已保存为 saved_model.h5")

# 加载模型
loaded_model = tf.keras.models.load_model('saved_model.h5')
print("模型已加载")

# 仅保存权重
model.save_weights('model_weights.h5')
print("权重已保存")

# 加载权重
model.load_weights('model_weights.h5')
print("权重已加载")

# 保存为SavedModel格式
model.save('saved_model', save_format='tf')
print("模型已保存为SavedModel格式")

# 清理
import os
for f in ['saved_model.h5', 'model_weights.h5', 'saved_model']:
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        import shutil
        shutil.rmtree(f)

print("已清理临时文件")

# TensorFlow Lite
print("\n=== TensorFlow Lite ===")

# 转换模型为TFLite格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存TFLite模型
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite模型已保存，大小: {len(tflite_model) / 1024:.2f} KB")

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# 获取输入输出信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"输入形状: {input_details[0]['shape']}")
print(f"输出形状: {output_details[0]['shape']}")

# 使用TFLite模型推理
test_sample = x_test_small[:1].astype('float32')
interpreter.set_tensor(input_details[0]['index'], test_sample)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite推理结果: {np.argmax(tflite_output)}")
print(f"原始模型预测: {np.argmax(model.predict(test_sample, verbose=0))}")

# 清理
if os.path.exists('model.tflite'):
    os.remove('model.tflite')
print("已清理TFLite模型文件")

# 分布式训练
print("\n=== 分布式训练策略 ===")

# MirroredStrategy - 多GPU训练
print("1. MirroredStrategy (多GPU同步训练):")
print("   - 适用于单机多GPU")
print("   - 同步训练，所有GPU同时更新")
print("   - 代码示例:")
print("""
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
    model.compile(...)
model.fit(...)
""")

# MultiWorkerMirroredStrategy
print("\n2. MultiWorkerMirroredStrategy (多机多GPU):")
print("   - 适用于多机多GPU")
print("   - 需要配置TF_CONFIG环境变量")

# ParameterServerStrategy
print("\n3. ParameterServerStrategy (参数服务器):")
print("   - 适用于大规模分布式训练")
print("   - 异步训练")

# 总结
print("\n=== TensorFlow高级特性学习总结 ===")
print("1. 张量操作和自动微分")
print("2. 自定义层和自定义模型")
print("3. 自定义训练循环")
print("4. 回调函数")
print("5. tf.data API数据管道")
print("6. 模型保存和加载")
print("7. TensorFlow Lite模型优化")
print("8. 分布式训练策略")

print("\nTensorFlow高级特性学习完成！")
