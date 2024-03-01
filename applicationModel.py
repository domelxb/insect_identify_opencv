import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 类别列表
categories = [
    "Apis-cerana-cerana-Fabricius",
    "Aromia-bungiii",
    "Chrysochroa-fulgidissima",
    "Chrysoperla-carnea",
    "Coccinella-septempunctata",
    "Conotrachelus-nenupha",
    "Empoasca-flavescens(Fab．)",
    "Graphocephala_coccinea",
    "Halyomorpha_halys",
    "Lyonetia_clerkella_L",
    "Myzus_persicae(Sulzer)",
    "Pterostichus_melanarius"
]

# 加载模型
model = load_model('models/cnn_best.h5')

def classify_image(image_path):
    # 加载图像
    img = image.load_img(image_path, target_size=(224, 224))  # 假设模型需要224x224大小的图像
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度

    # 预测
    predictions = model.predict(img_array)
    probabilities = predictions[0]

    # 创建类别与概率的映射
    return dict(zip(categories, probabilities))

def list_images(directory):
    # 列出所有图像文件
    images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 排序
    return images

# def save_results(results, txt_path, png_path):
#     # 保存结果到文本文件
#     with open(txt_path, 'w') as f:
#         for cls, prob in results.items():
#             f.write(f"{cls}: {prob:.6f}\n")
#
#     # 保存条形图为PNG
#     classes = list(results.keys())
#     probabilities = list(results.values())
#     plt.barh(classes, probabilities, color=['red' if 'C' in cls else 'blue' for cls in classes])
#     plt.xscale('log')
#     plt.xlabel('Probability')
#     plt.tight_layout()  # 确保标签不会被剪切
#     plt.savefig(png_path)
#     plt.close()
def save_results(results, txt_path, png_path, eps_path):
    # 保存结果到文本文件
    with open(txt_path, 'w') as f:
        for cls, prob in results.items():
            f.write(f"{cls}: {prob:.6f}\n")

    # 创建条形图
    classes = list(results.keys())
    probabilities = list(results.values())
    plt.barh(classes, probabilities, color=['red' if 'C' in cls else 'blue' for cls in classes])
    plt.xscale('log')
    plt.xlabel('Probability')
    plt.tight_layout()  # 确保标签不会被剪切

    # 保存条形图为PNG
    plt.savefig(png_path)
    # 保存条形图为EPS
    plt.savefig(eps_path)
    plt.close()

def main():
    directory = 'C:\\Users\\Administrator\\Desktop\\cv\\application-identify-image\\test-image'
    images = list_images(directory)

    if not images:
        print("No images found in the directory.")
        return

    for i, image in enumerate(images):
        print(f"{i+1}: {image}")

    choice = int(input("Select the image number to classify: "))
    selected_image = images[choice - 1]

    print(f"Classifying image: {selected_image}")
    results = classify_image(os.path.join(directory, selected_image))

    print("Classification results:")
    for cls, prob in results.items():
        print(f"{cls}: {prob:.6f}")

        # 这里是你添加的代码
    txt_path = os.path.join(directory, 'cnnApplication.txt')
    png_path = os.path.join(directory, 'cnnApplication.png')
    eps_path = os.path.join(directory, 'cnnApplication.eps')  # 新增EPS文件路径
    save_results(results, txt_path, png_path, eps_path)
    print(f"Results saved to {txt_path}, {png_path}, and {eps_path}")


if __name__ == "__main__":
    main()
