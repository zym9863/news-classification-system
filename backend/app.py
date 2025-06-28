from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import jieba
import json
import os
from datetime import datetime
import requests
import asyncio
import aiohttp

app = Flask(__name__)
CORS(app)

# 模型与文件路径（使用绝对路径，确保无论工作目录如何都能找到模型文件）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "my_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pickle")
STOPWORDS_PATH = os.path.join(BASE_DIR, "model", "停用词.txt")

from custom_layers import PositionalEmbedding, TransformerEncoder

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

# 加载模型和 tokenizer
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
    }
)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

stop_list  = pd.read_csv(STOPWORDS_PATH, index_col=False, quoting=3,  #停用词
                         sep="\t", names=['stopword'], encoding='utf-8')
#Jieba分词函数
def txt_cut(juzi):
    lis=[w for w in jieba.lcut(juzi) if w not in stop_list.values]
    return " ".join(lis)

# 类别标签字典
categories = {0: '教育', 1: '科技', 2: '社会', 3: '时政', 4: '财经', 5: '房产', 6: '家居'}

# 数据存储文件路径
HISTORY_FILE = "data/classification_history.json"
STATS_FILE = "data/classification_stats.json"
ACCURACY_FILE = "data/accuracy_records.json"

# 确保数据目录存在
os.makedirs("data", exist_ok=True)

# 初始化历史记录和统计数据
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {category: 0 for category in categories.values()}

def save_stats(stats):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def update_stats(category):
    stats = load_stats()
    stats[category] = stats.get(category, 0) + 1
    save_stats(stats)
    return stats

# 准确率记录相关函数
def load_accuracy_records():
    """加载准确率记录"""
    if os.path.exists(ACCURACY_FILE):
        with open(ACCURACY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_accuracy_records(records):
    """保存准确率记录"""
    with open(ACCURACY_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def get_ai_classification(text):
    """使用AI获取文本分类作为标准答案"""
    categories_list = "教育、科技、社会、时政、财经、房产、家居"
    prompt = f"""请对以下新闻文本进行分类，只从这些类别中选择一个：{categories_list}

新闻内容：{text}

请只返回分类名称，不要其他内容。"""
    
    try:
        ai_result = generate_ai_text_sync(prompt, "openai-large")
        # 清理AI返回结果，提取分类名称
        ai_category = ai_result.strip().replace('"', '').replace('。', '').replace('：', '').replace(':', '')
        
        # 验证分类是否在预定义列表中
        valid_categories = ['教育', '科技', '社会', '时政', '财经', '房产', '家居']
        for category in valid_categories:
            if category in ai_category:
                return category
        
        # 如果没有找到匹配的分类，返回None
        return None
    except Exception as e:
        print(f"AI分类失败: {str(e)}")
        return None

def calculate_accuracy():
    """计算当前模型的准确率"""
    records = load_accuracy_records()
    if not records:
        return 0.0
    
    correct_count = sum(1 for record in records if record['model_prediction'] == record['ai_standard'])
    total_count = len(records)
    
    return round((correct_count / total_count) * 100, 2) if total_count > 0 else 0.0

# Pollinations AI集成
async def generate_ai_text(prompt, model="openai-large"):
    """使用Pollinations API生成AI文本"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": model,
                "private": True
            }

            async with session.post(
                'https://text.pollinations.ai/',
                headers={'Content-Type': 'application/json'},
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return f"AI生成失败: HTTP {response.status}"
    except Exception as e:
        return f"AI生成错误: {str(e)}"

def generate_ai_text_sync(prompt, model="openai-large"):
    """同步版本的AI文本生成"""
    try:
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": model,
            "private": True
        }

        response = requests.post(
            'https://text.pollinations.ai/',
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.text
        else:
            return f"AI生成失败: HTTP {response.status_code}"
    except Exception as e:
        return f"AI生成错误: {str(e)}"

# 文本处理函数
def preprocess_text(text, tokenizer):
    """
    将文本转为模型可接受的输入格式：分词、序列化、填充
    :param text: 输入的文本
    :param tokenizer: 已加载的分词器（Tokenizer）
    :return: 处理后的序列化文本
    """
    text=txt_cut(text)
    # 文本转为序列
    sequences = tokenizer.texts_to_sequences([text])
    # 填充到固定长度
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
    return padded_sequences

# 类别预测函数
def predict_newkind(text, tokenizer, model):
    """
    根据输入文本预测类别
    :param text: 输入的文本
    :param tokenizer: 已加载的分词器（Tokenizer）
    :param model: 已加载的模型
    :return: 类别名称
    """
    # 文本预处理
    processed_text = preprocess_text(text, tokenizer)
    # 模型预测
    predictions = model.predict(processed_text)
    # 获取预测类别索引
    predicted_class = np.argmax(predictions, axis=1)
    # 返回对应的类别名称
    return categories[predicted_class[0]]

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "文本内容不能为空"}), 400

    # 调用预测函数
    predicted_category = predict_newkind(text, tokenizer, model)
    
    # 获取AI分类结果作为标准答案（异步进行，不影响用户体验）
    ai_category = None
    try:
        # 只对较短的文本进行AI验证（避免超时）
        if len(text) < 1000:
            ai_category = get_ai_classification(text)
    except Exception as e:
        print(f"AI分类验证失败: {str(e)}")

    # 保存到历史记录
    history = load_history()
    record = {
        "id": len(history) + 1,
        "text": text,
        "category": predicted_category,
        "timestamp": datetime.now().isoformat(),
        "text_length": len(text),
        "ai_standard": ai_category  # 添加AI标准答案
    }
    history.append(record)
    save_history(history)

    # 如果获得了AI分类结果，保存到准确率记录中
    if ai_category:
        accuracy_records = load_accuracy_records()
        accuracy_record = {
            "id": len(accuracy_records) + 1,
            "text": text[:200] + "..." if len(text) > 200 else text,  # 只保存前200字符
            "model_prediction": predicted_category,
            "ai_standard": ai_category,
            "is_correct": predicted_category == ai_category,
            "timestamp": datetime.now().isoformat()
        }
        accuracy_records.append(accuracy_record)
        save_accuracy_records(accuracy_records)

    # 更新统计数据
    stats = update_stats(predicted_category)

    # 计算当前准确率
    current_accuracy = calculate_accuracy()

    # 返回预测结果
    return jsonify({
        "category": predicted_category,
        "confidence": "高",  # 可以后续添加置信度计算
        "record_id": record["id"],
        "stats": stats,
        "ai_standard": ai_category,
        "accuracy": current_accuracy
    })

# 获取分类统计数据
@app.route('/stats', methods=['GET'])
def get_stats():
    stats = load_stats()
    total = sum(stats.values())

    # 计算百分比
    percentages = {}
    for category, count in stats.items():
        percentages[category] = round((count / total * 100) if total > 0 else 0, 2)

    # 计算准确率
    accuracy = calculate_accuracy()

    return jsonify({
        "stats": stats,
        "percentages": percentages,
        "total": total,
        "accuracy": accuracy
    })

# 获取历史记录
@app.route('/history', methods=['GET'])
def get_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    history = load_history()

    # 分页处理
    start = (page - 1) * per_page
    end = start + per_page
    paginated_history = history[start:end]

    return jsonify({
        "history": paginated_history,
        "total": len(history),
        "page": page,
        "per_page": per_page,
        "pages": (len(history) + per_page - 1) // per_page
    })

# AI文本生成接口
@app.route('/ai/generate', methods=['POST'])
def ai_generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    model = data.get('model', 'openai-large')

    if not prompt.strip():
        return jsonify({"error": "提示词不能为空"}), 400

    try:
        result = generate_ai_text_sync(prompt, model)
        return jsonify({
            "result": result,
            "prompt": prompt,
            "model": model
        })
    except Exception as e:
        return jsonify({"error": f"生成失败: {str(e)}"}), 500

# AI新闻摘要生成
@app.route('/ai/summarize', methods=['POST'])
def ai_summarize():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "文本内容不能为空"}), 400

    prompt = f"请为以下新闻内容生成一个简洁的摘要（100字以内）：\n\n{text}"

    try:
        summary = generate_ai_text_sync(prompt)
        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        })
    except Exception as e:
        return jsonify({"error": f"摘要生成失败: {str(e)}"}), 500

# AI新闻分析
@app.route('/ai/analyze', methods=['POST'])
def ai_analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "文本内容不能为空"}), 400

    prompt = f"""请分析以下新闻内容，提供以下信息：
1. 主要观点
2. 关键信息
3. 情感倾向
4. 重要性评级（1-5分）

新闻内容：
{text}"""

    try:
        analysis = generate_ai_text_sync(prompt)
        return jsonify({
            "analysis": analysis,
            "text_length": len(text)
        })
    except Exception as e:
        return jsonify({"error": f"分析失败: {str(e)}"}), 500

# 清除历史记录
@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    try:
        save_history([])
        save_stats({category: 0 for category in categories.values()})
        save_accuracy_records([])  # 同时清除准确率记录
        return jsonify({"message": "历史记录和统计数据已清除"})
    except Exception as e:
        return jsonify({"error": f"清除失败: {str(e)}"}), 500

# 获取准确率详细信息
@app.route('/accuracy', methods=['GET'])
def get_accuracy_details():
    """获取准确率详细信息"""
    records = load_accuracy_records()
    accuracy = calculate_accuracy()
    
    # 统计正确和错误的数量
    correct_count = sum(1 for record in records if record['is_correct'])
    total_count = len(records)
    
    # 按分类统计准确率
    category_accuracy = {}
    for category in categories.values():
        category_records = [r for r in records if r['model_prediction'] == category]
        if category_records:
            correct_in_category = sum(1 for r in category_records if r['is_correct'])
            category_accuracy[category] = {
                'total': len(category_records),
                'correct': correct_in_category,
                'accuracy': round((correct_in_category / len(category_records)) * 100, 2)
            }
        else:
            category_accuracy[category] = {'total': 0, 'correct': 0, 'accuracy': 0}
    
    return jsonify({
        "overall_accuracy": accuracy,
        "total_comparisons": total_count,
        "correct_predictions": correct_count,
        "category_accuracy": category_accuracy,
        "recent_records": records[-10:] if len(records) > 10 else records  # 最近10条记录
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)