import cv2
import os
from tqdm import tqdm
import numpy as np

def calculate_mse(img1, img2):
    # 计算均方误差 (MSE)
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')  # 如果没有误差，PSNR 无穷大
    max_pixel = 255.0  # 对于 8 位图像
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_psnrs(gt_folder, rendered_folder01, rendered_folder02):
    psnr_values01 = []
    psnr_values02 = []
    
    # 获取 gt_img 文件夹中的所有图像文件
    gt_images = os.listdir(gt_folder)
    
    for i in tqdm(range(0, 71)):
        for j in range(3):
            gt_img_path = os.path.join(gt_folder, f'{i:06d}_{j:01d}.png')
            rendered_img_path01 = os.path.join(rendered_folder01, f'{i:06d}_{j:01d}_rgb.png')
            rendered_img_path02 = os.path.join(rendered_folder02, f'frame_{i:04d}_{j:01d}.png')
            # print(gt_img_path)
            # print(rendered_img_path01)
            # print(rendered_img_path02)

            # 读取图像
            gt_img = cv2.imread(gt_img_path)
            rendered_img01 = cv2.imread(rendered_img_path01)
            rendered_img02 = cv2.imread(rendered_img_path02)

            # 检查图像是否成功读取
            if gt_img is None or rendered_img01 is None or rendered_img02 is None:
                print(f"无法读取图像")
                continue

            size_ours = (rendered_img01.shape[1], rendered_img01.shape[0])
            size_omnire = (rendered_img02.shape[1], rendered_img02.shape[0])
            gt_img01 = cv2.resize(gt_img, size_ours, interpolation=cv2.INTER_LINEAR)
            gt_img02 = cv2.resize(gt_img, size_omnire, interpolation=cv2.INTER_LINEAR)
            # rendered_img02 = cv2.resize(rendered_img02, size, interpolation=cv2.INTER_LINEAR)

            # 计算 PSNR
            psnr01 = calculate_psnr(gt_img01, rendered_img01)
            psnr02 = calculate_psnr(gt_img02, rendered_img02)
            psnr_values01.append(psnr01)
            psnr_values02.append(psnr02)

    return psnr_values01, psnr_values02

# 示例使用
gt_folder = '/nas/lys_data/data/waymo_23/processed/111/images'          # 替换为你的 gt_img 文件夹路径
rendered_folder1 = '/nas/lys_data/data/waymo_23/waymo_train_111/train/ours_45000'  # 替换为第一组渲染图像文件夹路径
rendered_folder2 = '/nas/lys_data/omnire/111/drivestudio/omnire/frames'  # 替换为第二组渲染图像文件夹路径

psnr01, psnr02 = calculate_psnrs(gt_folder, rendered_folder1, rendered_folder2)

print(f'Ours: {sum(psnr01)/len(psnr01)}')
print(f'Omnire: {sum(psnr02)/len(psnr02)}')
