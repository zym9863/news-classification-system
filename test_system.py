#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能新闻分类系统测试脚本
测试所有API接口的功能完整性
"""

import requests
import json
import time

# 配置
BASE_URL = "http://localhost:5000"
TEST_NEWS = [
    {
        "text": "教育部发布最新通知，要求各地加强中小学生心理健康教育工作，建立完善的心理健康服务体系。",
        "expected": "教育"
    },
    {
        "text": "苹果公司今日发布了最新的iPhone 15系列手机，采用了全新的A17芯片，性能相比上一代提升了20%。",
        "expected": "科技"
    },
    {
        "text": "央行今日宣布下调存款准备金率0.5个百分点，释放流动性约1万亿元。",
        "expected": "财经"
    }
]

def test_classification():
    """测试新闻分类功能"""
    print("🧪 测试新闻分类功能...")
    
    for i, news in enumerate(TEST_NEWS, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/classify",
                json={"text": news["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                category = result.get("category")
                print(f"  ✅ 测试 {i}: 预期={news['expected']}, 实际={category}")
                
                if category == news["expected"]:
                    print(f"     🎯 分类正确!")
                else:
                    print(f"     ⚠️  分类可能不准确")
            else:
                print(f"  ❌ 测试 {i} 失败: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ 测试 {i} 异常: {e}")
        
        time.sleep(1)  # 避免请求过快

def test_stats():
    """测试统计数据获取"""
    print("\n📊 测试统计数据功能...")
    
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"  ✅ 统计数据获取成功")
            print(f"     总数: {stats.get('total', 0)}")
            print(f"     分类统计: {stats.get('stats', {})}")
        else:
            print(f"  ❌ 统计数据获取失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ 统计数据获取异常: {e}")

def test_history():
    """测试历史记录功能"""
    print("\n📝 测试历史记录功能...")
    
    try:
        response = requests.get(f"{BASE_URL}/history", timeout=10)
        
        if response.status_code == 200:
            history = response.json()
            print(f"  ✅ 历史记录获取成功")
            print(f"     记录总数: {history.get('total', 0)}")
            print(f"     当前页记录: {len(history.get('history', []))}")
        else:
            print(f"  ❌ 历史记录获取失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ 历史记录获取异常: {e}")

def test_ai_generate():
    """测试AI文本生成"""
    print("\n🤖 测试AI文本生成功能...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai/generate",
            json={
                "prompt": "写一段关于人工智能发展的简短新闻",
                "model": "openai-large"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ AI文本生成成功")
            print(f"     生成长度: {len(result.get('result', ''))}")
            print(f"     使用模型: {result.get('model')}")
        else:
            print(f"  ❌ AI文本生成失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ AI文本生成异常: {e}")

def test_ai_summarize():
    """测试AI摘要生成"""
    print("\n📄 测试AI摘要功能...")
    
    test_text = """
    人工智能技术在近年来取得了突破性进展，特别是在自然语言处理、计算机视觉和机器学习等领域。
    深度学习算法的发展使得AI系统能够处理更复杂的任务，从语音识别到图像分析，再到自动驾驶汽车。
    然而，随着AI技术的快速发展，也带来了一些挑战，包括数据隐私、算法偏见和就业影响等问题。
    专家们认为，需要建立完善的AI治理框架，确保技术发展与社会责任并重。
    """
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai/summarize",
            json={"text": test_text.strip()},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ AI摘要生成成功")
            print(f"     原文长度: {result.get('original_length')}")
            print(f"     摘要长度: {result.get('summary_length')}")
            compression_ratio = (1 - result.get('summary_length', 0) / result.get('original_length', 1)) * 100
            print(f"     压缩率: {compression_ratio:.1f}%")
        else:
            print(f"  ❌ AI摘要生成失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ AI摘要生成异常: {e}")

def test_ai_analyze():
    """测试AI内容分析"""
    print("\n🔍 测试AI内容分析功能...")
    
    test_text = "央行今日宣布下调存款准备金率0.5个百分点，释放流动性约1万亿元。此举旨在支持实体经济发展，降低企业融资成本。"
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={"text": test_text},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ AI内容分析成功")
            print(f"     分析文本长度: {result.get('text_length')}")
            print(f"     分析结果长度: {len(result.get('analysis', ''))}")
        else:
            print(f"  ❌ AI内容分析失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ AI内容分析异常: {e}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("🚀 智能新闻分类系统功能测试")
    print("=" * 50)
    
    # 检查后端服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        print("✅ 后端服务运行正常\n")
    except:
        print("❌ 后端服务未运行，请先启动后端服务器")
        print("   运行命令: cd backend && python app.py")
        return
    
    # 执行各项测试
    test_classification()
    test_stats()
    test_history()
    test_ai_generate()
    test_ai_summarize()
    test_ai_analyze()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("=" * 50)
    print("\n💡 提示:")
    print("   - 如果AI功能测试失败，可能是网络连接问题")
    print("   - 分类准确性取决于模型训练质量")
    print("   - 建议在浏览器中访问 http://localhost:5173 查看完整界面")

if __name__ == "__main__":
    main()
