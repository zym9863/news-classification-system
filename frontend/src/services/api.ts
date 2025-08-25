/**
 * API服务配置
 * 用于与后端接口通信
 */
import axios from 'axios';
import type { CategoriesResponse, ModelInfoResponse, SinglePredictResponse, BatchPredictResponse } from '../types';

// 创建axios实例
const api = axios.create({
  // 在 Vite 中使用 import.meta.env 判断环境
  baseURL: import.meta.env.DEV ? 'http://localhost:8000/api' : '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
// 保留默认响应以便在调用处显式提取 data，避免类型不匹配
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

// API接口定义
export const newsApi = {
  // 获取所有类别
  getCategories: () => api.get<CategoriesResponse>('/categories').then(res => res.data),
  
  // 单条新闻分类预测
  predictSingle: (text: string) => api.post<SinglePredictResponse>('/predict', { text }).then(res => res.data),
  
  // 批量新闻分类预测
  predictBatch: (texts: string[]) => api.post<BatchPredictResponse>('/batch_predict', { texts }).then(res => res.data),
  
  // 获取模型信息
  getModelInfo: () => api.get<ModelInfoResponse>('/model_info').then(res => res.data),
};

export default api;