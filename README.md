# OSTrack

vitb_384_mae_ce_32x4_got10k_ep100: set train parameters

loader.py -> return torch.stack(batch, 1) in line 92

set:
lib/train/admin/local.py  # paths about training



# OSTrack 训练优化与 AMP 混合精度配置指南

> 本文档记录了在 RTX 5060 Ti 上训练 OSTrack（ViT-Base，GOT-10k）时，针对性能、稳定性和存储管理所做的优化配置，并包含混合精度训练（AMP）的原理与实施细节。适用于复现环境或作为项目 README 的技术补充。

------

## 一、背景与问题

- **原始状态**：使用 384 分辨率配置，`BATCH_SIZE=32`，无 AMP，单 Epoch 耗时约 51 分钟，FPS 仅约 3~5，显存不足引发内存交换。
- **目标配置**：分辨率 256，`BATCH_SIZE=40`，启用 AMP，`eps=1e-4`，保存间隔 10 个 Epoch。
- **遇到的问题**：
  - 中途开启 AMP 导致 IoU 变为 `NaN`（原因：优化器状态与 FP16 不兼容）。
  - YAML 中 `OPTIMIZER_ARGS` 未被代码识别，导致 `ValueError`。
  - `eps` 值被解析为字符串，引发类型错误。

------

## 二、优化方案总览

| 优化项                  | 作用                                    | 修改文件                          |
| :---------------------- | :-------------------------------------- | :-------------------------------- |
| 调整分辨率至 256        | 减少 Transformer Token 数量，降低计算量 | YAML 配置 (`SEARCH.SIZE: 256`)    |
| 增大 `BATCH_SIZE` 至 48 | 提高 GPU 利用率，减少迭代次数           | YAML 配置 (`BATCH_SIZE: 48`)      |
| 启用 AMP (`AMP: True`)  | 利用 Tensor Core 加速，降低显存         | YAML 配置 + `LTRTrainer` 传递参数 |
| 设置优化器 `eps=1e-4`   | 防止 FP16 下除零，稳定训练              | YAML + `base_functions.py`        |
| 保存间隔改为 10 Epoch   | 节省磁盘空间，便于管理                  | `base_trainer.py` 修改条件        |

------

## 三、详细修改步骤

### 1. 修改 YAML 配置文件

```
experiments/ostrack/vitb_256_mae_ce_32x4_got10k_ep100.yaml
```

```yaml
TRAIN:
  BATCH_SIZE: 48            # 原 40
  AMP: True                 # 启用混合精度
  OPTIMIZER_ARGS: {eps: 0.0001}   # 显式浮点数，避免解析为字符串
```

### 2. 注册 `OPTIMIZER_ARGS` 字段（`lib/config/ostrack/config.py`）

在 `cfg.TRAIN` 中添加：

```python
cfg.TRAIN.OPTIMIZER_ARGS = {}   # 默认空字典
```

### 3. 修改 `_update_config` 函数（同文件）

允许 `OPTIMIZER_ARGS` 直接赋值而不递归检查子键：



```python
def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    if k == 'OPTIMIZER_ARGS':
                        base_cfg[k] = v   # 直接赋值，不递归
                    else:
                        _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
```



### 4. 修改优化器创建函数（`lib/train/base_functions.py`）

在 `get_optimizer_scheduler` 中解包 `OPTIMIZER_ARGS` 并确保 `eps` 为浮点：



```python
def get_optimizer_scheduler(net, cfg):
    # ... 构建 param_dicts ...
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer_args = getattr(cfg.TRAIN, 'OPTIMIZER_ARGS', {})
        if 'eps' in optimizer_args:
            optimizer_args['eps'] = float(optimizer_args['eps'])
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                      **optimizer_args)
    # ...
```



### 5. 修改 Checkpoint 保存间隔（`lib/train/trainers/base_trainer.py`）

将保存条件中的 `epoch % 40 == 0` 改为 `epoch % 10 == 0`：

python

```python
if epoch > (max_epochs - 1) or save_every_epoch or epoch % 10 == 0 or epoch in save_epochs or epoch > (max_epochs - 5):
    # 保存 checkpoint
```



> **说明**：`save_epochs = [79, 159, 239]` 和最后 5 个 Epoch 的密集保存保留，便于获取最优模型。

### 6. 确认 AMP 开关已传递至 Trainer

`train_script.py` 中已有：

python

```python
use_amp = getattr(cfg.TRAIN, "AMP", False)
trainer = LTRTrainer(..., use_amp=use_amp)
```



无需额外修改。

------

## 四、AMP（自动混合精度）原理简述

- **核心思想**：在前向和反向传播中使用 `float16` 进行计算，利用 Tensor Core 加速矩阵乘加，同时保留 `float32` 的主权重和优化器状态以维持精度。
- **关键技术**：
  - `autocast`：自动将算子输入转换为合适精度（白名单算子用 FP16，黑名单如 Softmax 保留 FP32）。
  - `GradScaler`：动态缩放损失值，防止梯度下溢（FP16 最小表示约 6e-8）。
- **为何中途开启会 NaN**：优化器（AdamW）的动量和方差是在 FP32 下训练的，切换精度后新旧梯度尺度不匹配，导致更新量溢出 FP16 范围。
- **最佳实践**：从训练开始即启用 AMP，或加载已经用 AMP 训练过的 Checkpoint。

------

## 五、最终训练性能

| 指标              | 优化前（384, BS=32） | 优化后（256, BS=48, AMP） |
| :---------------- | :------------------- | :------------------------ |
| 单 Epoch 耗时     | ~51 分钟             | ~9.4 分钟                 |
| 显存占用          | ~88%（频繁交换）     | ~85%（稳定）              |
| FPS（训练）       | ~3-5                 | ~100+                     |
| Loss / IoU 稳定性 | 正常                 | 正常，无 NaN              |

训练日志示例（第 1 Epoch，迭代 800）：



```text
[train: 1, 800 / 1250] FPS: 106.3  ,  TotalTime: 0.451  ,  Loss/total: 5.77  ,  IoU: 0.468
```



------

## 六、注意事项

1. **切勿中途切换 AMP 状态**：如需启用，必须从头开始或从 AMP 训练过的 Checkpoint 恢复。
2. **`eps` 值调优**：默认 `1e-8` 在 FP16 下过于激进，推荐 `1e-4` ~ `1e-3`。
3. **Checkpoint 大小**：每个约 1.0~1.1 GB（含优化器状态），建议定期清理早期文件。
4. **Batch Size 调整**：以显存占用 85%~90% 为佳，过高会导致 OOM，过低则效率不足。

------

## 七、参考资料

- [PyTorch AMP 官方文档](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA 混合精度训练指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- 论文：*Mixed Precision Training* (Micikevicius et al., 2018)

------

> **总结**：通过以上配置，在 RTX 5060 Ti 上 OSTrack 训练达到效率与稳定性的平衡，所有修改均已测试通过，可放心应用于后续实验。

## 如何在我的Zeal集成NVIDIA的众多文档？NVIDIA文档如何快速搜索？

这些搜索结果提供了添加第三方文档集的一般方法。我的回答将涵盖几个方面：首先介绍Zeal中添加文档集的一般方法；然后说明如何获取NVIDIA文档的Docset；接着提供一些快速搜索NVIDIA文档的技巧；最后进行总结。

很遗憾，目前**NVIDIA官方并没有为Zeal提供可直接下载的Docset**。不过，我们仍可通过变通方法将NVIDIA文档集成到Zeal中，并利用一些技巧实现快速搜索。

### 🔧 方案一：利用社区资源，添加第三方Feed (最推荐)

这是最省力的方法，利用全球开发者共同维护的文档集仓库。

1. **获取Feed源**：访问 Zeal 第三方文档集仓库，例如 `https://zealusercontributions.now.sh/`或 `https://zealusercontributions.vercel.app/`。你可以在这里搜索 `NVIDIA`、`CUDA` 或 `TensorRT` 等关键词。
2. **复制链接**：如果找到对应的文档集，复制其 `.xml` 文件的链接地址。
3. **添加到Zeal**：打开Zeal，点击菜单栏 `Tools` -> `Docsets`，在 `Installed` 标签页中点击 **`Add feed`** 按钮。
4. **粘贴并下载**：在弹出的窗口中粘贴之前复制的XML链接，点击 `OK`。Zeal会自动识别并开始下载这个文档集。

> **注意**：社区维护的文档集可能不是官方最新版，或覆盖不全面。

### 🛠️ 方案二：自己动手，创建自定义Docset

如果找不到或想要最新、最完整的文档，可以自己动手丰衣足食。

1. **获取文档源**：访问NVIDIA官方文档网站（如 `docs.nvidia.com`），你需要一份结构化的HTML文档。如果NVIDIA提供了离线文档包（如PDF、CHM），可先下载。有些工具支持将CHM或HTML网页转换为Docset。
2. **准备转换工具**：你需要专门的工具将HTML文档转换为Docset格式。常见工具有：
   - **`doc2dash`**：一个Python工具，可以将HTML文档快速转为Docset。
   - **基于Ruby的生成器**：需要指定输入目录并配置索引规则。
3. **生成Docset文件**：根据工具指引，将HTML文档转换为一个包含索引文件（通常是SQLite数据库）的特定文件夹。
4. **导入Zeal**：
   - 打开Zeal，进入 `Tools` -> `Docsets`，在 `Installed` 标签页找到 **`Add local`** 或类似选项。
   - 选择你刚生成的Docset文件夹，Zeal会自动识别并添加。
   - 或者，直接将生成的 `.docset` 文件（本质上是一个文件夹）复制到Zeal的文档集目录中，然后重启Zeal。

### 🚀 NVIDIA文档快速搜索技巧

- **NVIDIA官方技术搜索**：关注“**NVIDIA英伟达企业解决方案**”微信公众号，在导航栏中找到“**技术搜索**”功能，输入问题即可快速检索。
- **官方API文档搜索**：NVIDIA的在线文档（如CUDA、DRIVE OS）通常自带搜索框。例如在CUDA文档页面，可直接在右上角的**Search**字段输入API名称进行快速检索。
- **Chrome扩展插件**：安装 **`CUDA Docs Switcher`** 扩展，可在不同版本CUDA文档间快速切换，并一键直达文档首页。
- **第三方文档平台**：使用 **`devdocs.io`**等聚合平台，它们支持离线使用并提供良好的搜索功能。
- **通用搜索技巧**：
  - **站点限定搜索**：在Google或Bing等搜索引擎中，使用 `site:nvidia.com <你的关键词>` 进行精准搜索。
  - **善用博客与GTC**：搜索 `NVIDIA Blog` 和 `GTC Talks` 加上相关主题，常能找到官方文档中未提及的深入讲解和最佳实践。

### 💎 总结

Zeal官方源中没有NVIDIA文档集，但通过**社区第三方Feed**或**自制Docset**的方式依然可以集成。建议先尝试方案一，如果不行再考虑方案二。
