import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 新闻分类接口
export const classifyText = async (text: string) => {
  const response = await api.post('/classify', { text })
  return response.data
}

// 获取统计数据
export const getStats = async () => {
  const response = await api.get('/stats')
  return response.data
}

// 获取历史记录
export const getHistory = async (page = 1, perPage = 10) => {
  const response = await api.get('/history', {
    params: { page, per_page: perPage }
  })
  return response.data
}

// AI文本生成
export const generateAIText = async (prompt: string, model = 'openai-large') => {
  const response = await api.post('/ai/generate', { prompt, model })
  return response.data
}

// AI摘要生成
export const generateSummary = async (text: string) => {
  const response = await api.post('/ai/summarize', { text })
  return response.data
}

// AI内容分析
export const analyzeContent = async (text: string) => {
  const response = await api.post('/ai/analyze', { text })
  return response.data
}

// 获取准确率详细信息
export const getAccuracyDetails = async () => {
  const response = await api.get('/accuracy')
  return response.data
}

// 清除历史记录
export const clearHistory = async () => {
  const response = await api.delete('/history/clear')
  return response.data
}

export default api
