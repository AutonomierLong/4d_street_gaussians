import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement  # 确保安装了 plyfile 库

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # 读取时间数据
    ply_data = PlyData.read(file_path)
    times = ply_data['vertex']['time']  # 假设时间数据存储在 vertex 中
    return points, colors, times

def storePly_with_time(path, xyz, rgb, time):
    # 设置 RGB 为 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)
    
    # 定义结构化数组的 dtype
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), 
             ('time', 'f4')]
    
    normals = np.zeros_like(xyz)

    if time.ndim == 1:
        time = time[:, np.newaxis]  # 转换为 2D 数组

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, time), axis=1)
    elements[:] = list(map(tuple, attributes))

    # 创建 PlyData 对象并写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def compute_density(points_xyz, radius=0.1):
    tree = KDTree(points_xyz)
    densities = np.zeros(len(points_xyz))

    for i, point in enumerate(points_xyz):
        neighbors = tree.query_ball_point(point, radius)  # 获取半径内的邻居
        densities[i] = len(neighbors)  # 计算邻居的数量

    return densities

def densify_point_cloud(points_xyz, points_rgb, points_time, num_new_points):
    tree = KDTree(points_xyz)
    densities = compute_density(points_xyz)
    
    # 计算反向权重
    density_probs = 1 / (densities + 1e-8)  # 添加小常数以防止除零
    density_probs /= density_probs.sum()  # 归一化权重
    
    new_points = []
    min_z = np.min(points_xyz[:, 2])
    max_z = np.max(points_xyz[:, 2])
    min_x = np.min(points_xyz[:, 0])
    max_x = np.max(points_xyz[:, 0])
    min_y = np.min(points_xyz[:, 1])
    max_y = np.max(points_xyz[:, 1])

    for _ in range(num_new_points):
        # 根据密度采样已有点
        sampled_index = np.random.choice(len(points_xyz), p=density_probs)
        sampled_point = points_xyz[sampled_index]

        while abs(sampled_point[2] - min_z) <= 0.025 or abs(sampled_point[2] - max_z) <= 0.025 or abs(sampled_point[1] - min_y) <= 0.025 or abs(sampled_point[1] - min_y) <= 0.025 or abs(sampled_point[0] - min_x) <= 0.025 or abs(sampled_point[0] - min_x) <= 0.025:
            sampled_index = np.random.choice(len(points_xyz), p=density_probs)
            sampled_point = points_xyz[sampled_index]
        
        # 在采样点周围生成新点
        perturbation = np.random.normal(scale=0.05, size=3)  # 添加噪声以生成新点
        new_point = sampled_point + perturbation
        
        distances, indices = tree.query(new_point, k=5)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()  # 归一化权重

        # 插值计算颜色和时间
        interpolated_rgb = np.dot(weights, points_rgb[indices])
        interpolated_time = np.dot(weights, points_time[indices])
        
        new_points.append((new_point, interpolated_rgb, interpolated_time))

    # 包含原始点
    for i in range(len(points_xyz)):
        new_points.append((points_xyz[i], points_rgb[i], points_time[i]))
    
    return np.array([p[0] for p in new_points]), np.array([p[1] for p in new_points]), np.array([p[2] for p in new_points])

# 设置输入输出路径
input_dir = '/nas/lys_data/data/waymo_23/waymo_train_447_time/input_ply/'
output_dir = '/nas/lys_data/data/waymo_23/waymo_train_447_time/input_ply/'  # 输出目录
num_new_points = 5000

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历目录中的文件
for file_name in os.listdir(input_dir):
    if 'obj' in file_name and file_name.endswith('.ply') and 'obj' in file_name:
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        
        points_xyz, points_rgb, points_time = read_ply(input_file)
        new_points_xyz, new_points_rgb, new_points_time = densify_point_cloud(points_xyz, points_rgb, points_time, num_new_points)
        storePly_with_time(output_file, new_points_xyz, new_points_rgb, new_points_time)

print("稠密化处理完成！")
