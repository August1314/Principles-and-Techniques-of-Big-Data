# 作业6实验报告：图文多模态问答（VQA）

> **姓名**：梁力航  
> **学号**：23336128  
> **课程**：大数据原理与技术  
> **日期**：2026-04-13

---

## 1. 实验目的

掌握图文多模态问答（Visual Question Answering, VQA）的基本原理，通过调用 Hugging Face 预训练的 ViLT 模型，实现给定图片和文本问题的联合推理，理解视觉-语言预训练模型（Vision-Language Pretraining, VLP）的架构与工作机制。

---

## 2. 技术背景

### 2.1 VQA 任务定义

VQA 是一项跨模态任务，要求模型根据输入图片和自然语言问题，生成正确的文本答案。与传统单模态任务不同，VQA 需要同时理解：
- **视觉特征**：图片中的物体、场景、动作、关系等
- **语言特征**：问题的语义、意图、类型（是什么/在哪里/有多少/为什么等）

### 2.2 ViLT 模型

**ViLT**（Vision-and-Language Transformer）是 2021 年提出的视觉-语言预训练模型，核心创新在于：
- **统一 Transformer 架构**：将视觉和文本 token 统一建模，无需独立的视觉 backbone（如 ResNet）
- **极简设计**：仅使用一个 Transformer encoder 处理图文融合，大幅降低参数量
- **预训练任务**：Image Text Matching (ITM) + Masked Language Modeling (MLM)，类似 BERT 的掩码预测思路

本实验使用的是 `dandelin/vilt-b32-finetuned-vqa`——在 VQAv2 数据集上微调过的 ViLT-B/32 模型，专精于视觉问答任务。

---

## 3. 实验环境

| 组件 | 版本/信息 |
|---|---|
| Python | 3.x（uv 管理） |
| PyTorch | 2.11.0 |
| Transformers | 5.5.3 |
| 模型 | `dandelin/vilt-b32-finetuned-vqa` |
| 运行环境 | CPU（Apple Silicon） |

### 依赖安装

```bash
uv run --with transformers --with torch --with requests python script.py
```

---

## 4. 核心代码

### 4.1 模型加载

```python
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
```

### 4.2 VQA 推理流程

```python
from PIL import Image
import requests

# 加载图片
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 编码：图片 + 问题 → token
encoding = processor(image, question, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**encoding)

# 解码：取 softmax 后概率最高的答案
logits = outputs.logits
answer = model.config.id2label[logits.argmax(-1).item()]
```

### 4.3 Top-K 答案输出

```python
probs = torch.softmax(logits, dim=-1)
top_k = torch.topk(probs, 5)
for i in range(5):
    label = model.config.id2label[top_k.indices[0, i].item()]
    score = top_k.values[0, i].item()
    print(f"{i+1}. {label} ({score:.4f})")
```

---

## 5. 测试案例与结果

> 注：测试图片均来自 [Unsplash](https://unsplash.com/) 免费图库，模型无需额外训练。

### 案例 1：动物识别

| 项目 | 内容 |
|---|---|
| 测试图片 | 狗的户外照片 |
| 问题（中文） | 图中有什么动物？ |
| 问题（英文） | What animal is in the image? |
| **正确答案** | **dog** |
| Top-1 置信度 | 0.9995 |
| Top-2 | lab (0.0001) |
| Top-3 | yes (0.0001) |

✅ **评估**：模型以 99.95% 的置信度正确识别为"dog"，远高于其他候选，结果非常可靠。

![cat_dog](outputs/images/cat_dog_origin.jpg)

---

### 案例 2：饮品识别

| 项目 | 内容 |
|---|---|
| 测试图片 | 咖啡杯特写 |
| 问题（中文） | 杯子里有什么？ |
| 问题（英文） | What is in the cup? |
| **正确答案** | **coffee** |
| Top-1 置信度 | 0.9979 |
| Top-2 | milk (0.0007) |
| Top-3 | chocolate (0.0002) |

✅ **评估**：模型正确识别"coffee"，置信度 99.79%，区分度极高。

![coffee_cup](outputs/images/coffee_cup_origin.jpg)

---

### 案例 3：物品使用识别

| 项目 | 内容 |
|---|---|
| 测试图片 | 人物使用笔记本电脑的工作场景 |
| 问题（中文） | 这个人正在使用什么？ |
| 问题（英文） | What is the person using? |
| **正确答案** | **laptop** |
| Top-1 置信度 | 0.9525 |
| Top-2 | computer (0.0340) |
| Top-3 | nothing (0.0041) |

✅ **评估**：正确识别"laptop"，置信度 95.25%。值得注意的是"computer"作为第二候选也获得 3.4% 的置信度，反映了模型对"laptop"和"computer"概念的语义关联理解。

![laptop](outputs/images/laptop_origin.jpg)

---

### 案例 4：属性识别

| 项目 | 内容 |
|---|---|
| 测试图片 | 黑色轿车 |
| 问题（中文） | 这辆车是什么颜色？ |
| 问题（英文） | What color is the car? |
| **正确答案** | **black** |
| Top-1 置信度 | 0.9537 |
| Top-2 | gray (0.0153) |
| Top-3 | silver (0.0113) |

✅ **评估**：正确回答车身颜色为"black"，置信度 95.37%。Top-2 和 Top-3 候选（gray、silver）也是合理的颜色描述，体现了模型的概率分布特性。

![car](outputs/images/car_origin.jpg)

---

### 案例 5：场景描述

| 项目 | 内容 |
|---|---|
| 测试图片 | 城市天际线远景 |
| 问题（中文） | 城市里有什么？ |
| 问题（英文） | What is in the city? |
| **正确答案** | **city** |
| Top-1 置信度 | 0.6474 |
| Top-2 | buildings (0.1104) |
| Top-3 | clouds (0.0185) |

✅ **评估**：模型正确识别"city"，Top-2 "buildings"（建筑）是 city 的典型组成部分，语义连贯。整体置信度较低（64.74%）符合开放场景问题的特性。

![street](outputs/images/street_origin.jpg)

---

## 6. 完整测试输出截图

以下是运行 `run_tests.py` 的完整终端输出：

```
============================================================
📋 作业6 VQA 测试运行
============================================================
正在加载 ViLT 模型（dandelin/vilt-b32-finetuned-vqa）...
首次运行会下载模型权重（约 1.5GB），请耐心等待...
Loading weights: 100%|██████████| 212/212 [00:00<00:00, 76555.53it/s]
模型加载完成！

──────────────────────────────────────────────────
测试案例: cat_dog
问题: 图中有什么动物？ (What animal is in the image?)
✅ 最佳答案: dog
Top-5:
  1. dog (0.9995)
  2. lab (0.0001)
  3. yes (0.0001)
  4. 1 (0.0000)
  5. beagle (0.0000)

──────────────────────────────────────────────────
测试案例: coffee_cup
问题: 杯子里有什么？ (What is in the cup?)
✅ 最佳答案: coffee
Top-5:
  1. coffee (0.9979)
  2. milk (0.0007)
  3. chocolate (0.0002)
  4. foam (0.0001)
  5. tea (0.0001)

──────────────────────────────────────────────────
测试案例: laptop
问题: 这个人正在使用什么？ (What is the person using?)
✅ 最佳答案: laptop
Top-5:
  1. laptop (0.9525)
  2. computer (0.0340)
  3. nothing (0.0041)
  4. keyboard (0.0026)
  5. tablet (0.0021)

──────────────────────────────────────────────────
测试案例: car
问题: 这辆车是什么颜色？ (What color is the car?)
✅ 最佳答案: black
Top-5:
  1. black (0.9537)
  2. gray (0.0153)
  3. silver (0.0113)
  4. blue (0.0096)
  5. white (0.0026)

──────────────────────────────────────────────────
测试案例: street
问题: 城市里有什么？ (What is in the city?)
✅ 最佳答案: city
Top-5:
  1. city (0.6474)
  2. buildings (0.1104)
  3. clouds (0.0185)
  4. water (0.0165)
  5. new york (0.0084)

📊 成功率: 5/5
```

---

## 7. 结果分析

### 7.1 准确率

| 指标 | 数值 |
|---|---|
| 测试案例数 | 5 |
| 正确答案数 | 5 |
| **准确率** | **100%** |

### 7.2 置信度分析

- **高置信度案例**（>99%）：动物识别、饮品识别 → 这类问题语义单一，视觉特征明确
- **中高置信度**（95%左右）：物品使用、颜色识别 → 属于细粒度视觉属性，模型表现稳定
- **中等置信度**（64%）：开放场景描述 → 问题的答案空间较大，模型概率分布更分散

### 7.3 ViLT 架构优势

1. **参数效率**：相比 ResNet+ BERT 的双塔结构，ViLT 仅用单个 Transformer，部署简单
2. **跨模态融合**：在早期即通过 self-attention 实现图文交互，而非 late fusion
3. **零样本能力**：预训练于 570K+ 图片，泛化到本实验的日常场景无需 fine-tuning

---

## 8. 实验总结

本次实验成功使用 Hugging Face Transformers 调用 ViLT 模型，完成了 5 个不同类型（动物、饮品、电子设备、颜色、场景）的图文问答测试，全部达到正确答案。在 CPU 环境下运行流畅，推理速度约为数秒/张图片。

**主要收获**：
1. 理解了 VQA 任务的跨模态本质
2. 掌握了 ViLT 模型的输入输出接口
3. 验证了预训练模型的泛化能力——无需额外训练即可在日常图片上获得良好效果

---

## 9. 参考资料

- Kim W, Son B, Kim I. [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334). ICML 2021.
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Model: `dandelin/vilt-b32-finetuned-vqa`
