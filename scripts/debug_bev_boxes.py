#!/usr/bin/env python3
"""
调试BEV可视化中目标框不显示的问题
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')
from unzipnuscenes import (
    load_lidar_points, visualize_bev_with_boxes, 
    draw_bev_box_2d, quaternion_to_rotation_matrix
)

def debug_bev_visualization():
    """调试BEV可视化中的目标框绘制"""
    
    nuscenes_root = '/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval'
    sample_idx = 0
    
    print("🔍 开始调试BEV目标框显示问题...")
    
    # 1. 加载注释文件
    annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
    
    with open(os.path.join(annotation_dir, 'sample.json'), 'r') as f:
        samples = json.load(f)
    
    with open(os.path.join(annotation_dir, 'sample_annotation.json'), 'r') as f:
        sample_annotations = json.load(f)
    
    with open(os.path.join(annotation_dir, 'category.json'), 'r') as f:
        categories = json.load(f)
    
    with open(os.path.join(annotation_dir, 'instance.json'), 'r') as f:
        instances = json.load(f)
    
    with open(os.path.join(annotation_dir, 'sample_data.json'), 'r') as f:
        sample_data = json.load(f)
    
    # 2. 获取样本数据
    sample = samples[sample_idx]
    sample_anns = [ann for ann in sample_annotations if ann['sample_token'] == sample['token']]
    
    print(f"📊 样本信息:")
    print(f"  - 样本token: {sample['token']}")
    print(f"  - 注释数量: {len(sample_anns)}")
    
    # 3. 分析注释数据
    print(f"\n📋 注释详细信息:")
    for i, ann in enumerate(sample_anns[:5]):  # 只显示前5个
        print(f"  注释 {i+1}:")
        print(f"    - Token: {ann['token']}")
        print(f"    - 位置: {ann['translation']}")
        print(f"    - 尺寸: {ann['size']}")
        print(f"    - 旋转: {ann['rotation']}")
        print(f"    - Instance token: {ann.get('instance_token', 'None')}")
        
        # 获取类别信息
        if ann.get('instance_token'):
            instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
            if instance and instance.get('category_token'):
                category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                if category:
                    print(f"    - 类别: {category['name']}")
        print()
    
    # 4. 加载ego pose数据
    with open(os.path.join(annotation_dir, 'ego_pose.json'), 'r') as f:
        ego_poses = json.load(f)
    
    # 5. 加载LiDAR数据
    lidar_data = next((sd for sd in sample_data 
                      if sd['sample_token'] == sample['token'] and 'LIDAR_TOP' in sd.get('filename', '')), None)
    
    if lidar_data:
        lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])
        print(f"📡 LiDAR文件: {lidar_path}")
        
        if os.path.exists(lidar_path):
            lidar_points = load_lidar_points(lidar_path)
            print(f"  - 点云数量: {len(lidar_points)}")
            
            # 6. 创建调试版本的BEV可视化
            print(f"\n🎨 创建调试BEV可视化...")
            debug_bev_with_detailed_info(lidar_points, sample_anns, categories, instances, ego_poses, lidar_data)
        else:
            print(f"❌ LiDAR文件不存在: {lidar_path}")
    else:
        print("❌ 未找到LiDAR数据")

def debug_bev_with_detailed_info(points, annotations, categories, instances, ego_poses, lidar_data):
    """创建带详细调试信息的BEV可视化"""
    
    # 获取ego pose信息进行坐标变换
    ego_pose_token = lidar_data['ego_pose_token']
    ego_pose = next((ep for ep in ego_poses if ep['token'] == ego_pose_token), None)
    
    if ego_pose:
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = np.array(ego_pose['rotation'])
        print(f"🚗 Ego pose - Translation: {ego_translation}, Rotation: {ego_rotation}")
    else:
        ego_translation = None
        ego_rotation = None
        print("⚠️  未找到ego pose信息")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 设置BEV范围
    bev_range = 50
    ax.set_xlim([-bev_range, bev_range])
    ax.set_ylim([-bev_range, bev_range])
    ax.set_aspect('equal')
    
    # 绘制LiDAR点云
    if len(points) > 0:
        valid_mask = ((np.abs(points[:, 0]) <= bev_range) & 
                     (np.abs(points[:, 1]) <= bev_range))
        points_bev = points[valid_mask]
        
        if len(points_bev) > 0:
            step = max(1, len(points_bev) // 20000)
            sampled_points = points_bev[::step]
            
            heights = sampled_points[:, 2]
            scatter = ax.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                               c=heights, cmap='terrain', s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Height (m)', shrink=0.8)
    
    # 调试目标框绘制
    boxes_drawn = 0
    boxes_in_range = 0
    category_counts = {}
    
    print(f"🔍 开始绘制目标框...")
    
    for i, ann in enumerate(annotations):
        # 获取类别名称
        category_name = "Unknown"
        if ann.get('instance_token'):
            instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
            if instance and instance.get('category_token'):
                category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                if category:
                    category_name = category['name']
        
        category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        # 获取3D框参数
        center = np.array(ann['translation'])
        size = np.array(ann['size'])
        rotation = np.array(ann['rotation'])
        
        print(f"  目标框 {i+1}: {category_name}")
        print(f"    原始位置: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"    尺寸: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]")
        print(f"    旋转: [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}]")
        
        # 应用坐标变换（从全局坐标转换为ego相对坐标）
        if ego_translation is not None and ego_rotation is not None:
            # 创建从全局到ego坐标的变换矩阵
            from unzipnuscenes import transform_matrix
            ego_transform = transform_matrix(ego_translation, ego_rotation, inverse=True)
            
            # 变换目标框中心到ego坐标系
            center_homogeneous = np.append(center, 1)
            center_ego = ego_transform @ center_homogeneous
            center_transformed = center_ego[:3]
            
            print(f"    变换后位置: [{center_transformed[0]:.2f}, {center_transformed[1]:.2f}, {center_transformed[2]:.2f}]")
            print(f"    距离ego车辆: {np.linalg.norm(center_transformed[:2]):.2f}m")
            
            # 检查是否在BEV范围内
            if abs(center_transformed[0]) <= bev_range and abs(center_transformed[1]) <= bev_range:
                boxes_in_range += 1
                print(f"    ✅ 在BEV范围内，开始绘制...")
                
                # 绘制增强的2D框
                try:
                    draw_bev_box_2d(ax, center_transformed, size, rotation, category_name)
                    boxes_drawn += 1
                    print(f"    ✅ 成功绘制目标框")
                except Exception as e:
                    print(f"    ❌ 绘制失败: {e}")
            else:
                print(f"    ⚠️  超出BEV范围 (距离: {np.linalg.norm(center_transformed[:2]):.2f}m)")
        else:
            print(f"    ⚠️  无ego pose信息，使用原始坐标")
            # 检查是否在BEV范围内
            if abs(center[0]) <= bev_range and abs(center[1]) <= bev_range:
                boxes_in_range += 1
                print(f"    ✅ 在BEV范围内，开始绘制...")
                
                # 绘制增强的2D框
                try:
                    draw_bev_box_2d(ax, center, size, rotation, category_name)
                    boxes_drawn += 1
                    print(f"    ✅ 成功绘制目标框")
                except Exception as e:
                    print(f"    ❌ 绘制失败: {e}")
            else:
                print(f"    ⚠️  超出BEV范围 (距离: {np.sqrt(center[0]**2 + center[1]**2):.2f}m)")
        print()
    
    # 添加ego车辆标记
    ax.plot(0, 0, 'ro', markersize=12, label='Ego Vehicle', markeredgecolor='white', markeredgewidth=3)
    ax.arrow(0, 0, 0, 5, head_width=2, head_length=2, fc='red', ec='white', linewidth=3)
    
    # 添加距离参考圆圈
    for radius in [10, 20, 30, 40]:
        if radius <= bev_range:
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.4, linestyle='--', linewidth=1.5)
            ax.add_patch(circle)
            # 添加距离标签
            ax.text(radius, 0, f'{radius}m', fontsize=10, ha='left', va='bottom', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 设置标题和标签
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    
    # 创建详细标题
    category_stats = ', '.join([f"{cat}: {count}" for cat, count in sorted(category_counts.items())])
    title = f'调试BEV可视化 - 总注释: {len(annotations)}, 范围内: {boxes_in_range}, 已绘制: {boxes_drawn}'
    if category_stats:
        title += f'\n类别统计: {category_stats}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # 保存调试图像
    output_path = './debug_bev_boxes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 调试BEV图像已保存: {output_path}")
    
    plt.close()
    
    # 打印总结
    print(f"\n📊 调试总结:")
    print(f"  - 总注释数量: {len(annotations)}")
    print(f"  - BEV范围内的目标框: {boxes_in_range}")
    print(f"  - 成功绘制的目标框: {boxes_drawn}")
    print(f"  - 类别分布: {category_counts}")

if __name__ == "__main__":
    debug_bev_visualization()