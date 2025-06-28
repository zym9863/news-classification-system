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

    # 保存到历史记录
    history = load_history()
    record = {
        "id": len(history) + 1,
        "text": text,
        "category": predicted_category,
        "timestamp": datetime.now().isoformat(),
        "text_length": len(text)
    }
    history.append(record)
    save_history(history)

    # 更新统计数据
    stats = update_stats(predicted_category)

    # 返回预测结果
    return jsonify({
        "category": predicted_category,
        "confidence": "高",  # 可以后续添加置信度计算
        "record_id": record["id"],
        "stats": stats
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

    return jsonify({
        "stats": stats,
        "percentages": percentages,
        "total": total
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
        return jsonify({"message": "历史记录已清除"})
    except Exception as e:
        return jsonify({"error": f"清除失败: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)