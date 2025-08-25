"""
新闻分类器模块
负责数据预处理、模型训练和预测功能
"""
import pandas as pd
import numpy as np
import jieba
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Dict, Any


class NewsClassifier:
    """新闻分类器类"""

    def __init__(
        self,
        data_path: str = "data.xlsx",
        text_column: str | None = None,
        label_column: str | None = None,
    ):
        """初始化分类器

        Args:
            data_path: 数据文件路径（支持 xlsx/xls/csv，默认工作目录下 data.xlsx）
            text_column: 文本列名（可选，若不填则自动解析常见别名）
            label_column: 标签列名（可选，若不填则自动解析常见别名）
        """
        self.data_path = data_path
        self.model = None
        self.vectorizer = None
        # 训练后从模型类目对齐，初始化仅作占位
        self.categories = ["教育", "科技", "社会", "时政", "财经", "房产", "家居"]
        self.model_path = "models/news_classifier.pkl"
        self.vectorizer_path = "models/vectorizer.pkl"

        # 可选的显式列名（优先）
        self._text_column = text_column
        self._label_column = label_column

        # 创建模型目录
        os.makedirs("models", exist_ok=True)

    # 内部工具：列名解析
    def _resolve_columns(self, df: pd.DataFrame) -> tuple[str, str]:
        """解析数据框中的文本列与标签列

        解析逻辑：
        - 若构造参数提供 text_column/label_column，直接校验并使用；
        - 否则在常见别名中（大小写不敏感）搜索；
        - 若仍未找到，尝试基于 dtype/列数进行启发式推断；
        - 找不到时抛出包含可用列名的清晰错误。

        Returns:
            (text_col, label_col): 实际存在于 df.columns 的列名二元组
        """
        cols = list(df.columns)
        normalized = {str(c).strip(): str(c).strip() for c in cols}
        lower_map = {str(c).strip().lower(): str(c).strip() for c in cols}

        # 优先使用显式列名
        if self._text_column:
            key = self._text_column.strip()
            if key in normalized:
                text_col = normalized[key]
            elif key.lower() in lower_map:
                text_col = lower_map[key.lower()]
            else:
                raise KeyError(
                    f"未找到文本列 '{self._text_column}'，可用列: {cols}"
                )
        else:
            text_aliases = [
                "标题", "题目", "新闻标题", "内容", "文本", "文章", "摘要",
                "title", "headline", "text", "content"
            ]
            text_col = next((lower_map[a] for a in (a.lower() for a in text_aliases) if a in lower_map), None)

        if self._label_column:
            key = self._label_column.strip()
            if key in normalized:
                label_col = normalized[key]
            elif key.lower() in lower_map:
                label_col = lower_map[key.lower()]
            else:
                raise KeyError(
                    f"未找到标签列 '{self._label_column}'，可用列: {cols}"
                )
        else:
            label_aliases = [
                "类别", "分类", "标签", "类目", "类型", "种类",
                "label", "category", "type", "class"
            ]
            label_col = next((lower_map[a] for a in (a.lower() for a in label_aliases) if a in lower_map), None)

        # 启发式兜底：若还没解析到
        if text_col is None or label_col is None:
            try:
                # 仅当列数为2时，推断第1列为文本、第2列为标签（常见简表）
                if (text_col is None or label_col is None) and len(cols) == 2:
                    c1, c2 = cols[0], cols[1]
                    # 优先将非数值型列作为文本列
                    if df[c1].dtype == object and text_col is None:
                        text_col = c1
                        label_col = label_col or c2
                    elif df[c2].dtype == object and text_col is None:
                        text_col = c2
                        label_col = label_col or c1
            except Exception:
                # 忽略启发式中的类型检查错误
                pass

        if text_col is None or label_col is None:
            # 构造详细错误消息
            raise KeyError(
                "无法解析文本/标签列名。可选解决方案:\n"
                f"1) 在初始化时显式传入列名: NewsClassifier(text_column='标题', label_column='类别')\n"
                f"2) 修改数据表列名为常见别名之一。当前可用列: {cols}"
            )

        print(f"列名解析: 文本列='{text_col}', 标签列='{label_col}'。所有列: {cols}")
        return text_col, label_col
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 中文分词
        words = jieba.cut(text)
        
        # 过滤长度小于2的词
        words = [word.strip() for word in words if len(word.strip()) >= 2]
        
        return " ".join(words)
    
    def load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            加载的数据框
        """
        try:
            # 支持简单的 csv/xlsx 自动选择
            if str(self.data_path).lower().endswith((".csv")):
                df = pd.read_csv(self.data_path)
            else:
                df = pd.read_excel(self.data_path)
            print(f"数据加载成功，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def train_model(self) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
            训练结果信息
        """
        print("开始训练模型...")
        # 加载数据
        df = self.load_data()
        # 解析列名并进行预处理
        text_col, label_col = self._resolve_columns(df)
        print("正在进行文本预处理...")
        df['processed_text'] = df[text_col].apply(self.preprocess_text)
        # 准备训练数据
        X = df['processed_text']
        y = df[label_col]
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # 文本向量化
        print("正在进行文本向量化...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        # 训练朴素贝叶斯模型
        print("正在训练朴素贝叶斯模型...")
        self.model = MultinomialNB(alpha=1.0)
        self.model.fit(X_train_vec, y_train)
        # 预测和评估
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型训练完成，准确率: {accuracy:.4f}")
        # 保存模型和向量化器
        self.save_model()
        # 生成分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        # 用训练得到的类目覆盖默认 categories，保持与模型 classes_ 对齐
        try:
            self.categories = list(getattr(self.model, 'classes_', [])) or self.categories
        except Exception:
            pass
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "categories": self.categories
        }
    
    def save_model(self) -> None:
        """保存训练好的模型和向量化器"""
        if self.model is not None and self.vectorizer is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            print("模型保存成功")
        else:
            print("没有训练好的模型可以保存")
    
    def load_model(self) -> bool:
        """
        加载训练好的模型
        
        Returns:
            是否加载成功
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                print("模型加载成功")
                return True
            else:
                print("模型文件不存在，需要先训练模型")
                # 自动训练模型
                self.train_model()
                return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        单条文本预测
        
        Args:
            text: 待分类的文本
            
        Returns:
            预测结果
        """
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise ValueError("模型未加载")
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 向量化
        text_vec = self.vectorizer.transform([processed_text])
        # 预测
        prediction = self.model.predict(text_vec)[0]
        prediction_proba = self.model.predict_proba(text_vec)[0]
        classes = list(getattr(self.model, 'classes_', [])) or self.categories
        # 获取最高置信度
        max_confidence = float(np.max(prediction_proba)) if hasattr(np, 'max') else float(max(prediction_proba))
        return {
            "category": prediction,
            "confidence": float(max_confidence),
            "all_probabilities": {
                category: float(prob)
                for category, prob in zip(classes, prediction_proba)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量文本预测
        
        Args:
            texts: 待分类的文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "category": "未知",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def get_categories(self) -> List[str]:
        """
        获取所有类别
        
        Returns:
            类别列表
        """
        return self.categories
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        if self.model is None:
            return {
                "status": "未加载",
                "categories": self.categories,
                "model_type": "MultinomialNB"
            }
        
        return {
            "status": "已加载",
            "categories": self.categories,
            "model_type": "MultinomialNB",
            "vectorizer_type": "TfidfVectorizer",
            "feature_count": self.vectorizer.max_features if self.vectorizer else None
        }


if __name__ == "__main__":
    # 训练和测试模型
    classifier = NewsClassifier()
    
    # 训练模型
    result = classifier.train_model()
    print("训练结果:", result)
    
    # 测试预测
    test_text = "教育部发布新的课程标准"
    prediction = classifier.predict(test_text)
    print(f"测试文本: {test_text}")
    print(f"预测结果: {prediction}")