/**
 * API服务配置
 * 用于与后端接口通信
 */
import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000/api',
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
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

// API接口定义
export const newsApi = {
  // 获取所有类别
  getCategories: () => api.get('/categories'),
  
  // 单条新闻分类预测
  predictSingle: (text: string) => api.post('/predict', { text }),
  
  // 批量新闻分类预测
  predictBatch: (texts: string[]) => api.post('/batch_predict', { texts }),
  
  // 获取模型信息
  getModelInfo: () => api.get('/model_info'),
};

export default api;