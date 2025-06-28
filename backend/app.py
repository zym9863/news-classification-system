from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import jieba

app = Flask(__name__)
CORS(app)

# 模型与文件路径
MODEL_PATH = "model\my_model.h5"
TOKENIZER_PATH = "model/tokenizer.pickle"

from custom_layers import PositionalEmbedding, TransformerEncoder

# 加载模型和 tokenizer
model = tf.keras.models.load_model(MODEL_PATH,custom_objects={"PositionalEmbedding": PositionalEmbedding,"TransformerEncoder": TransformerEncoder,})
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

stop_list  = pd.read_csv("model/停用词.txt",index_col=False,quoting=3,  #停用词
                         sep="\t",names=['stopword'], encoding='utf-8')
#Jieba分词函数
def txt_cut(juzi):
    lis=[w for w in jieba.lcut(juzi) if w not in stop_list.values]
    return " ".join(lis)

# 类别标签字典
categories = {0: '教育', 1: '科技', 2: '社会', 3: '时政', 4: '财经', 5: '房产', 6: '家居'}

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

    #调用预测函数
    predicted_category = predict_newkind(text, tokenizer, model)

    # 返回预测结果
    return jsonify({"category": predicted_category})

if __name__ == "__main__":
    app.run(debug=True, port=5000)