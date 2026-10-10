# Root Control 与早期任务适配器训练报告

历史版本核对  2026年10月9日  |  服务器 pku@100.94.2.67

本报告梳理原始 root control、早期 11 类任务适配器及其自由生成微调。核心结论是：前两项训练直接使用真实动作及测量得到的命令标签，没有专项教师蒸馏；自由生成微调没有专项教师，但使用冻结生成模型的输出作为保持参照。后来的 kick/jump 专项优化及多教师合并不属于这套早期方案。

### 三项训练的关系

| 阶段 | 更新对象 | 监督或参照 |
| --- | --- | --- |
| Root control | 独立 root adapter | 真实动作重建及根速度、转速 |
| 任务监督训练 | 一个共享 task adapter | 真实动作重建及任务命令量 |
| 自由生成微调 | 同一 task adapter | 命令误差及关闭任务控制后的生成参照 |

结构关系：冻结 FrankenMotion 底座 → 四层 Transformer；root adapter 和 task adapter 分别向这四层添加残差。两套 adapter 是分开的参数模块，早期 task adapter 内部由 11 类任务共享，不为每个动作单独加载一份网络。

### 训练了多少与实际采用多少

| 阶段 | 实际运行终点 | 选中检查点 |
| --- | --- | --- |
| Root control | epoch35 / 6965更新 | best.pt：epoch20 / 3980更新 |
| 任务监督训练 | 3300更新 | best.pt：step0 |
| 自由生成微调 | 8800更新 | free_best.pt：step7700 |

任务监督阶段的最优检查点是第0步，四个残差头末层保持零初始化。不能把该阶段描述为“经过3300步训练后获得有效命令控制”。后续自由生成微调从这个被选中的检查点开始。

### 底座版本

当时使用自行训练的 FrankenMotion 底座，冻结后训练 adapter。虽然目录名含 official，来源记录明确写为 Frozen user-trained backbone。后续计划使用官方发布权重是另一个实验版本，不能与本报告的历史结果混用。

## 原始 Root Control

root control 控制根部平面运动速度和身体朝向转速，不是直接指定世界坐标终点。输入随时间变化，因此可以给出分段速度或转速；其监督标签按1秒窗口平均。

### 数据与标签

数据审计记录12730条训练、1634条验证，来自现有 AMASS 运动特征及其标注划分。特征为20 Hz SMPL-RIFKE；根平面增量的模乘20得到速度，yaw增量乘20得到转速。以20帧为窗口平均，序列最后一个增量不参与有效标签统计。

每帧输入4维：速度除3 m/s、转速除π rad/s，以及这两个条件各自的有效位。训练时两个条件分别以15%概率丢弃，使网络学习仅速度、仅转速或缺失控制的情况。速度标签是非负速率，不能直接等同于任意方向的有符号速度。

### 网络与更新范围

共享编码器为4→128→512，SiLU激活；四个残差头各为512→128→512，分别接在四层Transformer输出后。末层权重和偏置零初始化，条件缺失时通过有效位关闭残差。可训练参数593536；底座完全冻结。

### 训练目标

对真实动作加扩散噪声，输入文字条件及从同一动作提取的根命令，预测干净动作。运动重建误差的前4个根特征权重为5，再加0.1倍速度MSE和0.1倍转速MSE。目标来自真实动作，不来自其他控制器的预测。

### 优化与检查点

AdamW学习率1e-4、weight decay 0.01、batch16、累积4次梯度，有效批量通常为64；梯度裁剪1。最多100个epoch、patience15。实际在epoch35提前停止，总计6965次更新；验证最优权重位于epoch20、3980次更新，验证损失0.123373。这个损失不是物理执行成功率。

### 监督类型

没有专项教师，没有任务专家路由，也不靠仿真奖励训练。真实动作既提供运动重建目标，也提供速度和转速标签。预训练底座提供先验，属于初始化依赖，不是这里的蒸馏教师。

## 早期共享任务 Adapter

该阶段在已有 root control 外增加一个共享任务网络；底座和 root control 都冻结，只训练 task adapter。覆盖抬手、前伸、出拳、挥手、转身、侧移、后退、踢腿、跳跃、前倾、前走11类。

### 真实动作来源与采样

从 frankenstein-local-available 标注中按全局caption关键词筛选动作，排除爬行、翻滚、躺下等词。每任务最多128条训练、16条验证；按source family去重，并排除与验证family相同的训练来源。实际共1403条训练、172条验证：strike为123/12，其余各128/16。

从片段开头固定裁取60或120帧；不足长度时保持最后姿态，并将新增帧的根增量清零。使用FK从实际裁剪动作计算命令量，缓存文字条件、运动特征和quantity。标签不要求天然落在指定命令范围内。

### 模型与输入

任务ID经11×32 embedding；任务数值按对应范围归一化后，与embedding拼成33维输入，再经33→128→512编码器及四个512→128→512残差头。参数597600，残差末层零初始化。这里的任务残差在同一序列内广播，没有后来版本增加的时间相位编码或独立205维输出残差头。

walk、back_walk、turn同时接入已有root条件，其余任务不传root控制。这是训练输入规则，不是切换任务专属权重。不同任务使用各自的quantity测量函数，也不等同于专项教师。

### 直接监督训练

真实动作加噪后，训练模型重建动作，并通过可微FK测量预测动作的命令量。损失为加权运动重建误差加0.2倍归一化命令Huber误差；前4个根特征重建权重为5。无需加载专项教师或对齐其他adapter的输出。

训练共3300步，学习率1e-4，AdamW，梯度裁剪1，seed2601003；每步一个样本，任务按11类循环，类内随机采样。验证使用固定噪声和固定扩散时刻t=50，每330步检查一次。

### 检查点实际结果与数据风险

best.pt选中step0，损失0.038167；训练终点损失0.310722。因此这次监督训练未在该验证准则下优于零残差起点。可能相关的因素包括关键词与实际时间窗口错位、首段未包含目标事件、补帧、命令覆盖不均；这些是需进一步验证的解释，不能当作已确定原因。

## 后续自由生成微调及方法边界

自由生成微调从监督阶段best.pt继续，保持底座和root adapter冻结，直接优化生成动作的参数响应。实际运行8800步，学习率3e-5，梯度裁剪1；训练通过10步可微DDIM反传，验证使用50步DDIM。最优free_best.pt位于7700步。

### 命令监督与生成参照

训练循环依次覆盖11类，随机选文本模板、在每类范围内均匀采样数值命令，并使用新的噪声。对相同文字和噪声，先关闭task adapter生成一份参照，再打开task adapter生成被训练的动作。需要root控制的任务，两条分支都保留root条件。

总目标包含：命令Huber误差，2倍相对姿态保持误差，0.002倍超过参照速度上限的惩罚，以及0.01倍支撑脚横向速度惩罚。支撑脚掩码由参照动作的脚高和脚速估计。

代码把这份生成参照命名为teacher。它没有单独训练的kick或jump专家，却仍是模型生成的保持目标。因此该阶段可以说“没有专项教师蒸馏”，不能说“只使用真实动作监督”或“完全没有模型参照”。协议提到冻结Sonic，但此训练损失没有调用Sonic或仿真rollout进行反传。

### 仍存在按任务设计的规则

姿态保持项对抬手、前伸、出拳、挥手的右臂关节降低权重，对kick的右腿关节降低权重。任务命令量的计算窗口和定义也不同。所有任务更新同一网络，但这并不等于没有任何任务专用损失规则。若新方案要求完全统一约束，这些规则需要重新设计。

### 评估边界

训练与测试共享每任务四个文本模板。该实验主要验证新噪声和数值命令，不能据此声称对未见文本泛化。最优自由生成验证分数0.095258，终点0.124127；该分数是命令误差与姿态误差组合，不等于G1物理执行成功率。

### 与后来的合并版本区分

后续11类统一蒸馏使用physical、kick、jump等教师；再后来的70000步合并训练使用root、task、goal、reach、exit等历史模块。它们不属于本报告的原始监督方案，不能把“早期没有专项教师”推广到最终aa3253e全链路。

### 对新方案的启示

允许真实动作参考时，可保留“从真实动作测量命令标签，再联合训练共享adapter”的思路。建议使用官方冻结底座、一个新初始化的共享命令网络、事件对齐的数据窗口和全任务均衡采样。取消历史教师、专项权重初始化和逐类单独微调；自由生成阶段是否保留基础模型参照应单独声明。

## 服务器资产与依据

服务器：pku@100.94.2.67。下列均为4090绝对路径。原始版本需要底座、配置、root与task分别加载，不是一个独立完整生成器文件。

### 底座与配置

```text
/home/pku/frankenmotion/outputs_amass/official_20260916/logs/checkpoints/last.ckpt
```

```text
/home/pku/frankenmotion/outputs_amass/root_control_20260925/base_config.yaml
```

这是历史自训底座；后续官方发布底座位于 pretrained/official_20260910/frankenmotion.ckpt，两者不可混用来复现本报告结果。

### Root 权重与训练依据

```text
/home/pku/frankenmotion/outputs_amass/root_control_20260925/best.pt
```

```text
/home/pku/frankenmotion/outputs_amass/root_control_20260925/code/train.py
```

```text
/home/pku/frankenmotion/outputs_amass/root_control_20260925/code/control.py
```

同目录下 provenance.json、data_audit.json、status.json、labels_train.jsonl、labels_val.jsonl记录底座、数据与训练终点。best.pt内的epoch和global_step用于确定实际采用步骤。

### 任务权重与训练依据

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/best.pt
```

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/free_best.pt
```

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/code/train_tasks.py
```

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/code/train_free.py
```

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/data_manifest.json
```

同目录cache/存放真实动作缓存，prompts/存放生成文本缓存；training_protocol.json、training_status.json、free_training_protocol.json、free_training_status.json保存训练协议和运行终点。

### 标注与运动特征

```text
/home/pku/frankenmotion/datasets/annotations/frankenstein-local-available/annotations/annotations.json
```

```text
/home/pku/frankenmotion/datasets/motions/AMASS_20.0_fps_nh_smplrifke
```

```text
/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/skeleton.npz
```

本报告核对了代码、运行协议和实际检查点元数据，没有重新训练或新增评测。最近下载的AMASS镜像和后来main三阶段方案不属于本历史实验的数据与结果。
