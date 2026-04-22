"""
航向角权重配置管理模块

支持在不同关联级别使用不同的权重策略:
- Level 1: 一级关联 (使用自适应权重)
- Level 1.5: 速度回溯关联 (不使用自适应权重，固定权重)
- Level 2+: 其他关联 (可选)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AngleWeightLevelConfig:
    """单个关联级别的角度权重配置"""
    
    enable: bool = True  # 是否启用角度权重
    use_adaptive: bool = True  # 是否使用自适应权重
    method: str = 'linear'  # 'sigmoid', 'linear', 'gaussian'
    base_weight: float = 0.15  # 基础权重
    
    def __repr__(self):
        return (f"AngleWeightLevelConfig("
                f"enable={self.enable}, "
                f"adaptive={self.use_adaptive}, "
                f"method={self.method}, "
                f"weight={self.base_weight})")


class AngleWeightLevelManager:
    """多级别航向角权重管理器"""
    
    def __init__(self):
        # Level 1: 一级关联 (使用自适应权重)
        self.level_1 = AngleWeightLevelConfig(
            enable=True,
            use_adaptive=True,
            method='linear',
            base_weight=0.15
        )
        
        # Level 1.5: 速度回溯关联 (不使用自适应权重)
        self.level_1_5 = AngleWeightLevelConfig(
            enable=True,
            use_adaptive=False,  # 关键：不使用自适应权重
            method='linear',
            base_weight=0.15
        )
        
        # Level 2+: 其他关联 (可选)
        self.level_2_plus = AngleWeightLevelConfig(
            enable=False,
            use_adaptive=False,
            method='linear',
            base_weight=0.15
        )
        
        self.verbose = False
    
    def get_config(self, level: str) -> AngleWeightLevelConfig:
        """获取指定级别的配置"""
        if level == 'level_1':
            return self.level_1
        elif level == 'level_1_5':
            return self.level_1_5
        elif level == 'level_2_plus':
            return self.level_2_plus
        else:
            raise ValueError(f"未知的关联级别: {level}")
    
    def set_level_1_method(self, method: str):
        """设置 Level 1 的权重方法"""
        self.level_1.method = method
        if self.verbose:
            print(f"[配置] Level 1 权重方法已设置为: {method}")
    
    def set_level_1_5_method(self, method: str):
        """设置 Level 1.5 的权重方法"""
        self.level_1_5.method = method
        if self.verbose:
            print(f"[配置] Level 1.5 权重方法已设置为: {method}")
    
    def enable_adaptive_for_level_1(self, enable: bool = True):
        """启用/禁用 Level 1 的自适应权重"""
        self.level_1.use_adaptive = enable
        if self.verbose:
            status = "启用" if enable else "禁用"
            print(f"[配置] Level 1 自适应权重已{status}")
    
    def disable_adaptive_for_level_1_5(self):
        """禁用 Level 1.5 的自适应权重（强制）"""
        self.level_1_5.use_adaptive = False
        if self.verbose:
            print(f"[配置] Level 1.5 自适应权重已禁用（强制）")
    
    def __repr__(self):
        return (f"AngleWeightLevelManager(\n"
                f"  Level 1: {self.level_1}\n"
                f"  Level 1.5: {self.level_1_5}\n"
                f"  Level 2+: {self.level_2_plus}\n"
                f")")


class ExperimentConfig:
    """实验配置管理"""
    
    def __init__(self):
        self.level_manager = AngleWeightLevelManager()
        self.experiment_name = "baseline"
        self.verbose = False
    
    def setup_experiment_1(self):
        """实验1: Linear 方法"""
        self.experiment_name = "exp1_linear"
        self.level_manager.set_level_1_method('linear')
        self.level_manager.set_level_1_5_method('linear')
        self.level_manager.disable_adaptive_for_level_1_5()
        
        if self.verbose:
            print(f"\n[实验] {self.experiment_name}")
            print(f"  Level 1: Linear + 自适应权重")
            print(f"  Level 1.5: Linear + 固定权重")
    
    def setup_experiment_2(self):
        """实验2: Gaussian 方法"""
        self.experiment_name = "exp2_gaussian"
        self.level_manager.set_level_1_method('gaussian')
        self.level_manager.set_level_1_5_method('gaussian')
        self.level_manager.disable_adaptive_for_level_1_5()
        
        if self.verbose:
            print(f"\n[实验] {self.experiment_name}")
            print(f"  Level 1: Gaussian + 自适应权重")
            print(f"  Level 1.5: Gaussian + 固定权重")
    
    def setup_experiment_3(self):
        """实验3: Sigmoid 方法"""
        self.experiment_name = "exp3_sigmoid"
        self.level_manager.set_level_1_method('sigmoid')
        self.level_manager.set_level_1_5_method('sigmoid')
        self.level_manager.disable_adaptive_for_level_1_5()
        
        if self.verbose:
            print(f"\n[实验] {self.experiment_name}")
            print(f"  Level 1: Sigmoid + 自适应权重")
            print(f"  Level 1.5: Sigmoid + 固定权重")
    
    def setup_baseline(self):
        """基线: 不使用角度权重"""
        self.experiment_name = "baseline_no_angle"
        self.level_manager.level_1.enable = False
        self.level_manager.level_1_5.enable = False
        
        if self.verbose:
            print(f"\n[实验] {self.experiment_name}")
            print(f"  Level 1: 不使用角度权重")
            print(f"  Level 1.5: 不使用角度权重")
    
    def __repr__(self):
        return (f"ExperimentConfig(\n"
                f"  实验名称: {self.experiment_name}\n"
                f"  {self.level_manager}\n"
                f")")


# 全局配置实例
_global_config = ExperimentConfig()


def get_global_config() -> ExperimentConfig:
    """获取全局配置"""
    return _global_config


def get_level_config(level: str) -> AngleWeightLevelConfig:
    """获取指定级别的配置"""
    return _global_config.level_manager.get_config(level)


def setup_experiment(experiment_name: str):
    """设置实验"""
    if experiment_name == 'exp1_linear':
        _global_config.setup_experiment_1()
    elif experiment_name == 'exp2_gaussian':
        _global_config.setup_experiment_2()
    elif experiment_name == 'exp3_sigmoid':
        _global_config.setup_experiment_3()
    elif experiment_name == 'baseline':
        _global_config.setup_baseline()
    else:
        raise ValueError(f"未知的实验: {experiment_name}")


def print_config():
    """打印当前配置"""
    print(_global_config)
