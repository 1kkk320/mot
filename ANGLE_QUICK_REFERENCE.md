# 航向角特征快速参考

## 一句话总结

**航向角特征在10%的复杂匹配场景中使用，贡献-8 IDSW，当前配置已达最优。**

---

## 核心配置（main.py）

```python
# 启用角度特征
tracker.tracker.enable_angle_in_level1 = True
tracker.tracker.angle_config.enable_angle_feature = True

# 关键参数（不要修改）
tracker.tracker.angle_config.angle_cost_sigma = 0.25  # σ值
tracker.tracker.angle_config.angle_weight = 0.45  # 角度权重
tracker.tracker.angle_config.angle_gate_threshold = math.radians(52)  # 门控52°
tracker.tracker.angle_config.gamma_min = 0.30  # 最小权重
tracker.tracker.appearance_weight_level1 = 0.10  # 外观权重
```

---

## 使用场景

### ✅ 使用角度特征（10%）
- **fused_cost路径**：多候选或IoU相近的复杂场景
- 权重分配：IoU(49.5%) > 角度(40.5%) > 外观(10%)

### ❌ 不使用角度特征（90%）
- **unique_iou路径**：IoU高且一对一的简单场景
- 直接基于IoU匹配，跳过角度特征

---

## 核心策略

### 1. 速度自适应权重
```
低速(<5 m/s)：角度权重30%（不稳定，降低权重）
中速(5-8 m/s)：角度权重30%-100%（指数过渡）
高速(>8 m/s)：角度权重100%（稳定，完全信任）
```

### 2. EMA平滑
```
低速(<5 m/s)：α=0.3（强平滑，减少噪声）
中速(5-8 m/s)：α=0.3-0.7（线性插值）
高速(>8 m/s)：α=0.7（快速响应，减少滞后）
```

### 3. 角度门控
- 拒绝角度差异 > 52度的候选
- 只在fused_cost路径中使用
- unique_iou路径不使用（保持稳定性）

### 4. 质量控制
- **阶段1（关联时）**：低质量检测降低角度权重
- **阶段2（关联后）**：低质量匹配不更新角度
- 质量阈值：0.5

---

## 性能贡献

```
基线（无角度）：IDSW = 134
启用角度特征：IDSW = 126
贡献：-8 IDSW（改善6.0%）
```

---

## 核心文件

```
tracking/angle_feature.py          # 角度特征核心
tracking/adaptive_angle_weight.py  # 自适应权重
tracking/angle_quality.py          # 质量评估
tracking/matching.py               # L1关联（路径选择）
tracking/cost_matrix_fusion.py     # 代价融合
tracking/tracker.py                # 角度更新
main.py                            # 配置参数
```

---

## 重要提示

### ⚠️ 不要修改参数
- 当前配置已经是最优的
- 任何调整都可能导致恶化
- 参数敏感性非常高

### ✅ 已验证的事实
- σ=0.25是最优值（0.30→IDSW+2，0.50→IDSW+4）
- 90%匹配走unique_iou路径是正常的
- 角度特征在10%场景中的贡献已达极限

### 🎯 下一步优化方向
- 停止优化角度特征
- 转向L1.5速度回溯（预期-3到-5 IDSW）
- 转向L2.5多帧回溯（预期-5到-8 IDSW）

---

## 调用流程（简化）

```
main.py → tracker.py → matching.py
                          ↓
                    判断路径
                    ├─→ unique_iou（90%）：不用角度
                    └─→ fused_cost（10%）：用角度
                          ↓
                    cost_matrix_fusion.py
                          ↓
                    angle_feature.py（计算代价）
                    adaptive_angle_weight.py（自适应权重）
                    angle_quality.py（质量评估）
                          ↓
                    返回融合代价
                          ↓
                    匈牙利算法匹配
                          ↓
                    tracker.py（角度更新+EMA平滑）
```

---

## 常见问题

### Q1: 为什么90%匹配不使用角度特征？
**A**: unique_iou路径是IoU高且一对一的简单场景，不需要额外特征。强行添加角度门控会破坏稳定性（测试显示IDSW+4）。

### Q2: 为什么不能提高σ值？
**A**: σ=0.25已经是最优值。提高到0.30会导致IDSW+2，提高到0.50会导致IDSW+4。

### Q3: 为什么角度特征贡献只有-8 IDSW？
**A**: KITTI数据集遮挡少，IoU已经很有效。角度特征只在10%复杂场景有用，这是数据集特点决定的。

### Q4: 还能继续优化角度特征吗？
**A**: 不建议。当前配置已达最优，继续优化可能适得其反。应该转向其他优化方向（L1.5, L2.5）。

---

## 版本历史

- **v1.0**（当前）：基线配置，IDSW=126，贡献-8
- 优化尝试：σ=0.30-0.50，angle_weight=0.55等，均导致恶化
- 结论：基线配置是最优的

---

## 联系与支持

如需修改配置或遇到问题，请参考：
- 详细文档：`ANGLE_STRATEGIES_IN_USE.md`
- 优化分析：`ANGLE_FINAL_CONCLUSION.md`
- 失败教训：`ANGLE_OPTIMIZATION_FAILURE_ANALYSIS.md`
