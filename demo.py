import os
import torch
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModel, AutoImageProcessor
from PIL import Image
from collections import Counter

# 模型和管道设置
repo_id = "openai/clip-vit-large-patch14-336"  # 用于 tokenizer 和 image_processor 的预训练模型
model_dir = './model/L-OpenCLIP-4e-6'  # 本地微调模型路径

# 加载预训练的 tokenizer 和 image_processor
image_processor = AutoImageProcessor.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id, config=AutoConfig.from_pretrained(repo_id))
model = AutoModel.from_pretrained(model_dir)

# 设置设备为 GPU（若无 GPU 则用 CPU）
device = 0 if torch.cuda.is_available() else -1

# 创建管道
clip_pipeline = pipeline(
    model=model,
    task="zero-shot-image-classification",
    tokenizer=tokenizer,
    device=device,
    image_processor=image_processor,
    config=AutoConfig.from_pretrained(model_dir)
)


# 获取图像路径和标签
def get_image_paths(main_folder_path):
    subfolders = [d for d in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, d))]
    image_paths = []
    labels = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_paths.append(os.path.join(subfolder_path, filename))
                labels.append(subfolder)
    return image_paths, labels, subfolders


# 批处理生成器
def batch_generator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# 测试函数
def test_dataset(main_folder_path, dataset_name, batch_size=32):
    print(f"\n=== Testing {dataset_name} ===")

    # 获取图像路径和标签
    image_paths, labels, subfolders = get_image_paths(main_folder_path)
    print(f"Found {len(image_paths)} images in {dataset_name}")

    all_predictions = []
    all_true_labels = []
    errors = []

    # 组合图像路径和标签
    image_paths_and_labels = list(zip(image_paths, labels))

    # 分批处理图像
    for batch in batch_generator(image_paths_and_labels, batch_size):
        batch_images = []
        batch_labels = []
        batch_filenames = []
        for image_path, label in batch:
            try:
                with Image.open(image_path) as img:
                    batch_images.append(img.copy())
                batch_labels.append(label)
                batch_filenames.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
        if not batch_images:
            continue

        # 获取预测
        predictions = clip_pipeline(images=batch_images, candidate_labels=subfolders)
        predicted_labels = [pred[0]['label'] for pred in predictions]

        all_predictions.extend(predicted_labels)
        all_true_labels.extend(batch_labels)

        # 收集错误
        for true_label, predicted_label, filename in zip(batch_labels, predicted_labels, batch_filenames):
            if true_label != predicted_label:
                errors.append((true_label, predicted_label, filename))

        # 清理批次图像以释放内存
        batch_images.clear()

    # 计算总体准确率
    correct_counts = Counter()
    incorrect_counts = Counter()

    for true_label, predicted_label in zip(all_true_labels, all_predictions):
        if true_label == predicted_label:
            correct_counts[true_label] += 1
        else:
            incorrect_counts[true_label] += 1

    total_correct = sum(correct_counts.values())
    total_incorrect = sum(incorrect_counts.values())
    total_accuracy = (total_correct / (total_correct + total_incorrect)) * 100 if (total_correct + total_incorrect) > 0 else 0
    print(f"\nTotal Accuracy for {dataset_name}: {total_accuracy:.2f}%")



# 数据集路径（假设下载到 data 文件夹）
bcmd_path = "./data/Test_BCMD"
cmmd_path = "./data/Test_CMMD"

# 测试两个数据集
if __name__ == "__main__":
    test_dataset(bcmd_path, "Test_BCMD")
    test_dataset(cmmd_path, "Test_CMMD")