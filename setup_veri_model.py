#!/usr/bin/env python3
"""
VeRi模型配置脚本
用于设置和验证VeRi预训练模型
"""

import os
import shutil
from pathlib import Path
import torch

def setup_veri_model():
    """设置VeRi模型"""
    
    print("🚗 VeRi车辆ReID模型配置脚本")
    print("=" * 50)
    
    # 检查模型文件
    model_path = Path("external/weights/veri_sbs_R50-ibn.pth")
    
    if not model_path.exists():
        print("❌ 未找到VeRi模型文件")
        print(f"请将下载的模型文件放置到: {model_path}")
        print("\n📋 操作步骤:")
        print("1. 确保你已下载 veri_sbs_R50-ibn.pth 模型文件")
        print("2. 将文件复制到 external/weights/ 目录")
        print("3. 重新运行此脚本")
        return False
    
    print(f"✅ 找到VeRi模型文件: {model_path}")
    
    # 检查模型文件大小
    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"📊 模型文件大小: {file_size:.1f} MB")
    
    # 验证模型可以加载
    try:
        print("🔍 验证模型文件...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model' in checkpoint:
            print("✅ 模型格式正确 (包含 'model' 键)")
        elif 'state_dict' in checkpoint:
            print("✅ 模型格式正确 (包含 'state_dict' 键)")
        else:
            print("⚠️  模型格式可能不标准，但会尝试加载")
            
        print(f"📋 模型信息:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if key != 'model' and key != 'state_dict':
                    print(f"   {key}: {checkpoint[key]}")
                    
    except Exception as e:
        print(f"❌ 模型文件验证失败: {e}")
        return False
    
    # 检查配置文件
    config_path = Path("external/fast_reid/configs/VeRi/sbs_R50-ibn.yml")
    if config_path.exists():
        print(f"✅ 找到VeRi配置文件: {config_path}")
    else:
        print(f"❌ 未找到VeRi配置文件: {config_path}")
        return False
    
    # 检查适配器文件
    adaptor_path = Path("external/adaptors/fastreid_veri_adaptor.py")
    if adaptor_path.exists():
        print(f"✅ VeRi适配器已创建: {adaptor_path}")
    else:
        print(f"❌ VeRi适配器未找到: {adaptor_path}")
        return False
    
    # 检查新的EmbeddingComputer
    embedding_path = Path("trackers/integrated_ocsort_embedding/embedding_veri.py")
    if embedding_path.exists():
        print(f"✅ VeRi EmbeddingComputer已创建: {embedding_path}")
    else:
        print(f"❌ VeRi EmbeddingComputer未找到: {embedding_path}")
        return False
    
    print("\n🎉 VeRi模型配置完成！")
    print("\n📈 预期性能提升:")
    print("   • Rank@1: 97.0% (车辆识别准确率)")
    print("   • mAP: 81.9% (平均精度)")
    print("   • 专门针对车辆优化的特征提取")
    print("   • 比通用ReID模型更适合车辆跟踪")
    
    print("\n🚀 下一步:")
    print("   运行: python main.py --config configs/your_config.yaml")
    print("   系统将自动使用VeRi预训练模型")
    
    return True

def test_veri_model():
    """测试VeRi模型加载"""
    print("\n🧪 测试VeRi模型加载...")
    
    try:
        from external.adaptors.fastreid_veri_adaptor import FastReIDVeRi
        
        # 创建模型实例
        model = FastReIDVeRi()
        print("✅ VeRi模型加载成功")
        
        # 测试推理
        dummy_input = torch.randn(1, 3, 384, 128).cuda().half()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ 推理测试成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   特征维度: {output.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ VeRi模型测试失败: {e}")
        return False

if __name__ == "__main__":
    success = setup_veri_model()
    
    if success:
        # 如果有GPU，测试模型加载
        if torch.cuda.is_available():
            test_veri_model()
        else:
            print("⚠️  未检测到CUDA，跳过模型加载测试")
    else:
        print("\n❌ 配置失败，请检查上述问题后重试")