# 消融实验结果记录

## 实验进度

| 实验 | 配置 | IDSW | 变化 | IDF1 | MOTA | HOTA | 状态 |
|------|------|------|------|------|------|------|------|
| 1. 基线 | 无优化 | 134 | - | 87.642% | 84.204% | 79.2 | ✅ 完成 |
| 2. +角度 | +角度特征 | 126 | -8 | 87.675% | 84.221% | 79.235 | ✅ 完成 |
| 3. +L1.5 | +角度+L1.5 | ? | ? | ? | ? | ? | ⏸️ 待测试 |
| 4. +L2.5 | +角度+L1.5+L2.5 | ? | ? | ? | ? | ? | ⏸️ 待测试 |
| 5. 完整 | 全部功能 | 128 | -6 | ? | ? | ? | ✅ 已知 |

---

## 实验1: 基线（无优化）

**配置**:
```python
velocity_backtrack_enabled = False  # ❌ L1.5速度回溯
enable_angle_in_level1 = False      # ❌ 角度特征
enable_multi_frame_backtrack = False # ❌ L2.5多帧回溯
use_acceleration_gate = False       # ❌ 加速度门控
```

**完整结果**:
```
HOTA   DetA   AssA   DetRe  DetPr  AssRe  AssPr  LocA   MOTA   IDF1   IDSW
79.2   77.916 80.724 83.315 88.998 83.905 92.825 91.427 84.204 87.642 134
```

**分析**: 纯基线性能，作为对比基准

---

## 实验2: 基线 + 角度特征

**配置**:
```python
velocity_backtrack_enabled = False  # ❌ L1.5速度回溯
enable_angle_in_level1 = True       # ✅ 角度特征
enable_multi_frame_backtrack = False # ❌ L2.5多帧回溯
use_acceleration_gate = False       # ❌ 加速度门控

# 角度特征参数
angle_level1_weight = 0.35
angle_level1_method = 'gaussian'
angle_level1_sigma = 0.35
angle_level1_gate_threshold = math.radians(60)
```

**完整结果**:
```
HOTA   DetA   AssA   DetRe  DetPr  AssRe  AssPr  LocA   MOTA   IDF1   IDSW
79.235 77.921 80.789 83.323 88.995 83.976 92.829 91.427 84.221 87.675 126
```

**详细指标**:
- HOTA: 79.235
- DetA: 77.921
- AssA: 80.789
- DetRe: 83.323
- DetPr: 88.995
- AssRe: 83.976
- AssPr: 92.829
- LocA: 91.427
- RHOTA: 82.035
- MOTA: 84.221%
- MOTP: 90.739
- IDF1: 87.675%
- IDR: 84.882%
- IDP: 90.659%
- IDSW: 126
- MT: 442
- PT: 92
- ML: 30
- Frag: 254

**分析**:
- ✅ IDSW从134降到126（降低8个，改善6.0%）
- ✅ IDF1从87.642%提升到87.675%（+0.033%）
- ✅ MOTA从84.204%提升到84.221%（+0.017%）
- ✅ HOTA从79.2提升到79.235（+0.035）
- ✅ 所有指标都略有提升或保持稳定
- ✅ 没有负面影响

**角度特征贡献**: -8 IDSW（6.0%改善）

**结论**: 角度特征有效，贡献显著，接近之前的最佳效果（-10）

---

## 实验3: 基线 + 角度特征 + L1.5速度回溯（待测试）

**配置**:
```python
velocity_backtrack_enabled = True   # ✅ L1.5速度回溯
enable_angle_in_level1 = True       # ✅ 角度特征
enable_multi_frame_backtrack = False # ❌ L2.5多帧回溯
use_acceleration_gate = False       # ❌ 加速度门控
```

**预期**: IDSW应该降低到110-120左右

---

## 实验4: 基线 + 角度 + L1.5 + L2.5（待测试）

**配置**:
```python
velocity_backtrack_enabled = True   # ✅ L1.5速度回溯
enable_angle_in_level1 = True       # ✅ 角度特征
enable_multi_frame_backtrack = True # ✅ L2.5多帧回溯
use_acceleration_gate = False       # ❌ 加速度门控
```

**预期**: IDSW应该降低到100-110左右

---

## 实验5: 完整配置（已知）

**配置**:
```python
velocity_backtrack_enabled = True   # ✅ L1.5速度回溯
enable_angle_in_level1 = True       # ✅ 角度特征
enable_multi_frame_backtrack = True # ✅ L2.5多帧回溯
use_acceleration_gate = True        # ✅ 加速度门控
```

**已知结果**: IDSW = 128

**注意**: 这个结果比实验2（IDSW=126）还要差，说明L1.5+L2.5+加速度门控的组合可能存在问题，或者参数需要调优。

---

## 功能贡献分析

基于当前数据：

| 功能 | IDSW贡献 | 占比 |
|------|----------|------|
| 角度特征 | -8 | 100% (当前) |
| L1.5 + L2.5 + 加速度 | ? | 待测试 |

**疑问**: 为什么完整配置（实验5，IDSW=128）比只有角度特征（实验2，IDSW=126）还要差2个？

**可能原因**:
1. L1.5/L2.5/加速度门控的参数不合适
2. 多个功能之间存在负面交互
3. 需要逐步测试每个功能的独立贡献

---

## 下一步

**选项1**: 继续优化角度特征，争取达到-10的效果
- 尝试方案B（稳健优化）
- 调整参数: weight=0.40, sigma=0.30, gate=55°

**选项2**: 接受当前角度特征效果，继续实验3
- 启用L1.5速度回溯
- 观察L1.5的独立贡献

**推荐**: 选项2，继续消融实验，找出为什么完整配置效果反而变差

---

**更新时间**: 2026-04-15
**当前状态**: 实验2完成，角度特征贡献-8 IDSW
**下一步**: 实验3（+L1.5速度回溯）
