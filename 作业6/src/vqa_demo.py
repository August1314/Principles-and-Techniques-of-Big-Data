#!/usr/bin/env python3
"""
作业6：图文多模态问答（VQA）
使用 Hugging Face ViLT 预训练模型
"""
import os
import sys

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import torch


def load_model():
    """加载 ViLT 模型和处理器"""
    print("正在加载 ViLT 模型（dandelin/vilt-b32-finetuned-vqa）...")
    print("首次运行会下载模型权重（约 1.5GB），请耐心等待...")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    print("模型加载完成！")
    return processor, model


def answer_question(processor, model, image_path: str, question: str, device="cpu"):
    """
    对图片提问，返回答案
    
    Args:
        image_path: 图片路径（本地或 URL）
        question: 问题文本
        device: 设备（cpu/cuda）
    """
    # 加载图片
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    
    # 编码
    encoding = processor(image, question, return_tensors="pt")
    
    # 移到设备
    for k, v in encoding.items():
        encoding[k] = v.to(device)
    
    # 推理
    model.to(device)
    with torch.no_grad():
        outputs = model(**encoding)
    
    # 解码答案（取 top-5）
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    
    # 取 top-k 答案
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.topk(probs, 5)
    
    results = []
    for i in range(5):
        label = model.config.id2label[top_k.indices[0, i].item()]
        score = top_k.values[0, i].item()
        results.append((label, score))
    
    return answer, results


def interactive_demo(processor, model):
    """交互式演示"""
    print("\n" + "=" * 60)
    print("📷  VQA 交互式演示（输入 q 退出）")
    print("=" * 60)
    
    while True:
        image_path = input("\n图片路径（本地路径或 URL，输入 q 退出）: ").strip()
        if image_path.lower() == "q":
            print("退出演示。")
            break
        
        question = input("问题: ").strip()
        if not question:
            print("问题不能为空！")
            continue
        
        try:
            answer, top5 = answer_question(processor, model, image_path, question)
            print(f"\n✅ 最佳答案: {answer}")
            print("Top-5 答案:")
            for i, (label, score) in enumerate(top5, 1):
                print(f"  {i}. {label} ({score:.4f})")
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    processor, model = load_model()
    
    # 如果指定了图片和问题，直接运行
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        question = sys.argv[2]
        answer, top5 = answer_question(processor, model, image_path, question)
        print(f"\n✅ 最佳答案: {answer}")
        print("Top-5 答案:")
        for i, (label, score) in enumerate(top5, 1):
            print(f"  {i}. {label} ({score:.4f})")
    else:
        interactive_demo(processor, model)
