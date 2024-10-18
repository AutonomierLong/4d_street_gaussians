import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    times = np.random.rand(len(points))  # 模拟时间数据
    return points, colors, times

def write_ply(file_path, points, colors, times):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_path, pcd)

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

    for _ in range(num_new_points):
        # 根据密度采样已有点
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

# 使用示例
input_file = '/nas/lys_data/data/waymo_23/waymo_train_447_time/input_ply/points3D_obj_123.ply'
output_file = 'output.ply'
num_new_points = 1000

points_xyz, points_rgb, points_time = read_ply(input_file)
new_points_xyz, new_points_rgb, new_points_time = densify_point_cloud(points_xyz, points_rgb, points_time, num_new_points)
write_ply(output_file, new_points_xyz, new_points_rgb, new_points_time)
