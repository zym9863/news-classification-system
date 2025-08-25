/**
 * 类型定义文件
 * 定义应用中使用的数据类型
 */

// 新闻类别类型
export type NewsCategory = '教育' | '科技' | '社会' | '时政' | '财经' | '房产' | '家居';

// 预测结果类型
export interface PredictionResult {
  category: NewsCategory;
  confidence: number;
  all_probabilities?: Record<string, number>;
  error?: string;
}

// 单条预测请求的响应
export interface SinglePredictResponse {
  text: string;
  predicted_category: NewsCategory;
  confidence: number;
}

// 批量预测响应
export interface BatchPredictResponse {
  results: PredictionResult[];
}

// 类别列表响应
export interface CategoriesResponse {
  categories: NewsCategory[];
}

// 模型信息响应
export interface ModelInfoResponse {
  status: string;
  categories: NewsCategory[];
  model_type: string;
  vectorizer_type?: string;
  feature_count?: number;
}

// 新闻条目类型（用于展示）
export interface NewsItem {
  id: string;
  text: string;
  predictedCategory?: NewsCategory;
  confidence?: number;
  isProcessing?: boolean;
}