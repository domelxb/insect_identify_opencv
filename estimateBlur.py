import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def estimate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.Laplacian(image, cv2.CV_64F).var()

def process_folder(folder_path):
    sharpness_values = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            sharpness = estimate_sharpness(file_path)
            if sharpness is not None:
                sharpness_values.append((filename, sharpness))
    return sharpness_values

# 实际文件夹路径
folder1_path = 'C:/Users/Administrator/Desktop/cv/split_data/train/test-1'
folder2_path = 'C:/Users/Administrator/Desktop/cv/split_data/train/test-2'

# 处理两个文件夹
folder1_sharpness = process_folder(folder1_path)
folder2_sharpness = process_folder(folder2_path)

# 结果保存路径
results_dir = 'C:/Users/Administrator/Desktop/cv/insect-identify/results/'

# 检查并创建结果目录
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 保存结果到文本文件
with open(os.path.join(results_dir, 'result.txt'), 'w') as file:
    file.write('Test-1 Folder:\n')
    for name, sharpness in folder1_sharpness:
        file.write(f'{name}: {sharpness}\n')
    file.write('\nTest-2 Folder:\n')
    for name, sharpness in folder2_sharpness:
        file.write(f'{name}: {sharpness}\n')

# 计算平均值和标准差
mean_values = [np.mean([sharpness for name, sharpness in folder_sharpness]) for folder_sharpness in [folder1_sharpness, folder2_sharpness]]
std_dev_values = [np.std([sharpness for name, sharpness in folder_sharpness]) for folder_sharpness in [folder1_sharpness, folder2_sharpness]]

# 制作柱状图
labels = ['Test-1', 'Test-2']
x_pos = np.arange(len(labels))

fig, ax = plt.subplots()
ax.bar(x_pos, mean_values, yerr=std_dev_values, align='center', alpha=0.7, ecolor='black', capsize=10)

# 添加数据点
for i, folder_sharpness in enumerate([folder1_sharpness, folder2_sharpness]):
    y_values = [sharpness for name, sharpness in folder_sharpness]
    ax.scatter([i] * len(y_values), y_values, alpha=0.7)

ax.set_xlabel('Folder')
ax.set_ylabel('Sharpness Value')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Comparison of Image Sharpness with Data Points and Error Bars')

# 保存图像
plt.savefig(os.path.join(results_dir, 'result.png'))
plt.savefig(os.path.join(results_dir, 'result.eps'), format='eps')

plt.show()
