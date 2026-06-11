# OpenCV人脸识别学习
# 主要内容：人脸检测、人脸识别、人脸特征点、人脸对齐

# 导入必要的库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载人脸检测器
print("=== 加载人脸检测器 ===")

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

print("人脸检测器加载完成")

# 创建示例图像
print("\n=== 创建示例图像 ===")

# 创建一个简单的人脸图像
face_img = np.zeros((300, 300, 3), dtype=np.uint8)

# 绘制人脸轮廓
cv2.ellipse(face_img, (150, 150), (80, 100), 0, 0, 360, (255, 200, 150), -1)

# 绘制眼睛
cv2.circle(face_img, (120, 130), 15, (0, 0, 0), -1)
cv2.circle(face_img, (180, 130), 15, (0, 0, 0), -1)
cv2.circle(face_img, (123, 127), 5, (255, 255, 255), -1)
cv2.circle(face_img, (183, 127), 5, (255, 255, 255), -1)

# 绘制嘴巴
cv2.ellipse(face_img, (150, 180), (20, 15), 0, 0, 360, (0, 0, 0), 2)

# 绘制眉毛
cv2.line(face_img, (105, 115), (135, 112), (0, 0, 0), 2)
cv2.line(face_img, (165, 112), (195, 115), (0, 0, 0), 2)

print("示例人脸图像创建完成")

# 显示图像
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
plt.title('示例人脸图像')
plt.axis('off')
plt.show()

# 人脸检测
print("\n=== 人脸检测 ===")

# 转换为灰度图像
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"检测到的人脸数量: {len(faces)}")

# 在图像上绘制人脸框
img_with_faces = face_img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"人脸位置: x={x}, y={y}, w={w}, h={h}")

# 显示结果
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
plt.title('人脸检测结果')
plt.axis('off')
plt.show()

# 检测眼睛
print("\n=== 眼睛检测 ===")

img_with_eyes = face_img.copy()
gray = cv2.cvtColor(img_with_eyes, cv2.COLOR_BGR2GRAY)

# 在人脸区域内检测眼睛
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img_with_eyes[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(f"检测到的眼睛数量: {len(eyes)}")
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

# 显示结果
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_eyes, cv2.COLOR_BGR2RGB))
plt.title('眼睛检测结果')
plt.axis('off')
plt.show()

# 检测微笑
print("\n=== 微笑检测 ===")

img_with_smile = face_img.copy()
gray = cv2.cvtColor(img_with_smile, cv2.COLOR_BGR2GRAY)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img_with_smile[y:y+h, x:x+w]
    
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    print(f"检测到的微笑数量: {len(smiles)}")
    
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

# 显示结果
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_smile, cv2.COLOR_BGR2RGB))
plt.title('微笑检测结果')
plt.axis('off')
plt.show()

# 使用DNN人脸检测器
print("\n=== 使用DNN人脸检测器 ===")

# 加载DNN模型
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

# 检查文件是否存在，如果不存在则创建简单示例
import os
if not os.path.exists(modelFile) or not os.path.exists(configFile):
    print("DNN模型文件不存在，使用Haar检测器进行演示")
else:
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    
    # 准备输入图像
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (300, 300), [104, 177, 123])
    
    # 设置输入
    net.setInput(blob)
    
    # 前向传播
    detections = net.forward()
    
    # 处理检测结果
    img_dnn = face_img.copy()
    h, w = face_img.shape[:2]
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(img_dnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(f"DNN检测到人脸，置信度: {confidence:.4f}")
    
    # 显示结果
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_dnn, cv2.COLOR_BGR2RGB))
    plt.title('DNN人脸检测结果')
    plt.axis('off')
    plt.show()

# 人脸特征点检测
print("\n=== 人脸特征点检测 ===")

# 使用dlib进行人脸特征点检测（需要安装dlib）
try:
    import dlib
    
    # 加载人脸检测器和特征点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # 检测人脸
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb)
    
    print(f"检测到的人脸数量: {len(dets)}")
    
    # 获取特征点
    img_landmarks = face_img.copy()
    for det in dets:
        shape = predictor(img_rgb, det)
        
        # 绘制特征点
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(img_landmarks, (x, y), 2, (0, 255, 0), -1)
    
    # 显示结果
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2RGB))
    plt.title('人脸68个特征点')
    plt.axis('off')
    plt.show()
    
except ImportError:
    print("dlib未安装，跳过特征点检测示例")
except FileNotFoundError:
    print("shape_predictor_68_face_landmarks.dat文件未找到，跳过特征点检测示例")

# 人脸对齐
print("\n=== 人脸对齐 ===")

# 创建人脸对齐函数
def align_face(image, landmarks):
    """根据特征点对齐人脸"""
    # 眼睛中心
    left_eye_center = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                               (landmarks.part(37).x, landmarks.part(37).y),
                               (landmarks.part(38).x, landmarks.part(38).y),
                               (landmarks.part(39).x, landmarks.part(39).y),
                               (landmarks.part(40).x, landmarks.part(40).y),
                               (landmarks.part(41).x, landmarks.part(41).y)], axis=0)
    
    right_eye_center = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)], axis=0)
    
    # 计算眼睛中心连线的角度
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 90
    
    # 计算缩放比例
    desired_right_eye_x = 1.0 - 0.35
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - 0.35) * 150
    scale = desired_dist / dist
    
    # 计算旋转中心
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # 更新平移参数
    tX = 150 * 0.5
    tY = 150 * 0.4
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    # 应用变换
    aligned_face = cv2.warpAffine(image, M, (150, 150), flags=cv2.INTER_CUBIC)
    
    return aligned_face

try:
    if 'shape' in locals():
        aligned_face = align_face(face_img, shape)
        
        # 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        axes[1].set_title('对齐后图像')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"人脸对齐失败: {e}")

# 人脸识别（简单示例）
print("\n=== 人脸识别 ===")

# 创建人脸数据库
print("创建人脸数据库...")

# 生成一些示例人脸特征（模拟）
face_database = {
    '张三': np.random.rand(128),
    '李四': np.random.rand(128),
    '王五': np.random.rand(128)
}

# 计算人脸特征（模拟）
def extract_features(image):
    """提取人脸特征（模拟）"""
    return np.random.rand(128)

# 人脸识别函数
def recognize_face(image, database, threshold=0.6):
    """人脸识别"""
    features = extract_features(image)
    
    min_distance = float('inf')
    best_match = None
    
    for name, db_features in database.items():
        distance = np.linalg.norm(features - db_features)
        if distance < min_distance:
            min_distance = distance
            best_match = name
    
    if min_distance < threshold:
        return best_match, min_distance
    else:
        return "未知", min_distance

# 测试人脸识别
name, distance = recognize_face(face_img, face_database)
print(f"识别结果: {name}, 距离: {distance:.4f}")

# 人脸验证
print("\n=== 人脸验证 ===")

def verify_face(image1, image2, threshold=0.6):
    """验证两张人脸是否属于同一个人"""
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    
    distance = np.linalg.norm(features1 - features2)
    is_same = distance < threshold
    
    return is_same, distance

# 创建另一张人脸图像
face_img2 = face_img.copy()
# 稍微修改一下（模拟不同光照）
face_img2 = cv2.add(face_img2, np.array([10, 10, 10], dtype=np.uint8))

is_same, distance = verify_face(face_img, face_img2)
print(f"人脸验证结果: {'相同' if is_same else '不同'}, 距离: {distance:.4f}")

# 人脸识别流程总结
print("\n=== 人脸识别流程总结 ===")
print("1. 人脸检测：使用Haar级联或DNN检测器")
print("2. 人脸对齐：根据特征点进行对齐")
print("3. 特征提取：使用预训练模型提取特征")
print("4. 人脸比对：计算特征距离进行识别/验证")

# 创建多人脸图像
print("\n=== 多人脸检测 ===")

# 创建包含多个人脸的图像
multi_face_img = np.zeros((400, 600, 3), dtype=np.uint8)

# 添加多个人脸
faces_data = [
    {'pos': (100, 100), 'scale': 1.0},
    {'pos': (300, 80), 'scale': 0.8},
    {'pos': (480, 120), 'scale': 1.2},
    {'pos': (150, 280), 'scale': 0.9},
    {'pos': (350, 260), 'scale': 1.1}
]

for face in faces_data:
    x, y = face['pos']
    scale = face['scale']
    # 绘制人脸
    cv2.ellipse(multi_face_img, (x, y), (int(60*scale), int(80*scale)), 0, 0, 360, (255, 200, 150), -1)
    cv2.circle(multi_face_img, (int(x-25*scale), int(y-20*scale)), int(12*scale), (0, 0, 0), -1)
    cv2.circle(multi_face_img, (int(x+25*scale), int(y-20*scale)), int(12*scale), (0, 0, 0), -1)

# 检测多人脸
gray = cv2.cvtColor(multi_face_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"检测到 {len(faces)} 个人脸")

# 绘制人脸框
img_multi_faces = multi_face_img.copy()
for i, (x, y, w, h) in enumerate(faces, 1):
    cv2.rectangle(img_multi_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_multi_faces, f'人脸{i}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(img_multi_faces, cv2.COLOR_BGR2RGB))
plt.title('多人脸检测结果')
plt.axis('off')
plt.show()

print("\nOpenCV人脸识别学习完成！")
