#!/usr/bin/env python3
"""
生成 VQA 测试可视化截图（用于实验报告）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from vqa_demo import load_model, answer_question
import os
from PIL import Image
import requests
import io

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试案例
TEST_CASES = [
    {
        "name": "cat_dog",
        "image": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "question": "What animal is in the image?",
        "question_cn": "图中有什么动物？",
    },
    {
        "name": "coffee_cup",
        "image": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=400",
        "question": "What is in the cup?",
        "question_cn": "杯子里有什么？",
    },
    {
        "name": "laptop",
        "image": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400",
        "question": "What is the person using?",
        "question_cn": "这个人正在使用什么？",
    },
    {
        "name": "car",
        "image": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",
        "question": "What color is the car?",
        "question_cn": "这辆车是什么颜色？",
    },
    {
        "name": "street",
        "image": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=400",
        "question": "What is in the city?",
        "question_cn": "城市里有什么？",
    },
]


def download_image(url):
    """下载图片"""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def generate_screenshot(processor, model):
    """生成可视化报告截图"""
    print("正在生成可视化截图...")
    
    for tc in TEST_CASES:
        print(f"处理: {tc['name']}")
        img = download_image(tc['image'])
        
        answer, top5 = answer_question(processor, model, tc['image'], tc['question'])
        
        # 保存图片
        img.save(os.path.join(OUTPUT_DIR, "images", f"{tc['name']}_origin.jpg"))
        
        # 下载并保存带标注的图片（简单复制）
        img.save(os.path.join(OUTPUT_DIR, "images", f"{tc['name']}_with_answer.jpg"))
        
        print(f"  → {tc['question_cn']} → {answer}")

    print(f"\n所有图片已保存至: {os.path.join(OUTPUT_DIR, 'images')}")


if __name__ == "__main__":
    processor, model = load_model()
    generate_screenshot(processor, model)
