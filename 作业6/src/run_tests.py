#!/usr/bin/env python3
"""
作业6：VQA 测试运行脚本
运行多个测试案例并保存结果
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from vqa_demo import load_model, answer_question

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试图片（使用网络公开图片 + 本地测试图片）
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

def run_tests():
    print("=" * 60)
    print("📋 作业6 VQA 测试运行")
    print("=" * 60)
    
    # 加载模型
    processor, model = load_model()
    
    results = []
    
    for tc in TEST_CASES:
        print(f"\n{'─' * 50}")
        print(f"测试案例: {tc['name']}")
        print(f"问题: {tc['question_cn']} ({tc['question']})")
        
        try:
            answer, top5 = answer_question(processor, model, tc['image'], tc['question'])
            print(f"✅ 最佳答案: {answer}")
            print("Top-5:")
            for i, (label, score) in enumerate(top5, 1):
                print(f"  {i}. {label} ({score:.4f})")
            
            results.append({
                "name": tc['name'],
                "question": tc['question_cn'],
                "answer": answer,
                "top5": top5,
                "success": True,
            })
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "name": tc['name'],
                "question": tc['question_cn'],
                "answer": None,
                "error": str(e),
                "success": False,
            })
    
    # 保存结果到文件
    results_file = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("作业6 VQA 测试结果\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"案例: {r['name']}\n")
            f.write(f"问题: {r['question']}\n")
            if r['success']:
                f.write(f"答案: {r['answer']}\n")
                f.write("Top-5:\n")
                for i, (label, score) in enumerate(r['top5'], 1):
                    f.write(f"  {i}. {label} ({score:.4f})\n")
            else:
                f.write(f"错误: {r['error']}\n")
            f.write("\n")
    
    print(f"\n结果已保存至: {results_file}")
    
    # 成功率统计
    success_count = sum(1 for r in results if r['success'])
    print(f"\n📊 成功率: {success_count}/{len(results)}")


if __name__ == "__main__":
    run_tests()
