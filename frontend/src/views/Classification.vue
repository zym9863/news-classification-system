<template>
  <div class="classification-container">
    <el-card class="main-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Document /></el-icon>
          <span>新闻文本分类</span>
        </div>
      </template>

      <div class="input-section">
        <el-form :model="form" label-width="100px">
          <el-form-item label="新闻内容">
            <el-input
              v-model="form.text"
              type="textarea"
              :rows="8"
              placeholder="请输入要分类的新闻内容..."
              maxlength="2000"
              show-word-limit
            />
          </el-form-item>
          <el-form-item>
            <el-button 
              type="primary" 
              @click="classifyText"
              :loading="loading"
              :disabled="!form.text.trim()"
              size="large"
            >
              <el-icon><MagicStick /></el-icon>
              开始分类
            </el-button>
            <el-button @click="clearText" size="large">
              <el-icon><Delete /></el-icon>
              清空
            </el-button>
            <el-button @click="loadExample" size="large">
              <el-icon><Document /></el-icon>
              示例文本
            </el-button>
          </el-form-item>
        </el-form>
      </div>

      <!-- 分类结果 -->
      <div v-if="result" class="result-section">
        <el-divider content-position="left">分类结果</el-divider>
        <el-alert
          :title="`分类结果：${result.category}`"
          type="success"
          :description="`置信度：${result.confidence} | 记录ID：${result.record_id}`"
          show-icon
          :closable="false"
        />
        
        <!-- 快速操作 -->
        <div class="quick-actions">
          <el-button type="primary" @click="generateSummary" :loading="summaryLoading">
            <el-icon><Document /></el-icon>
            生成摘要
          </el-button>
          <el-button type="success" @click="analyzeContent" :loading="analysisLoading">
            <el-icon><Search /></el-icon>
            内容分析
          </el-button>
          <el-button type="info" @click="viewStats">
            <el-icon><PieChart /></el-icon>
            查看统计
          </el-button>
        </div>
      </div>

      <!-- AI摘要结果 -->
      <div v-if="summary" class="ai-result-section">
        <el-divider content-position="left">AI智能摘要</el-divider>
        <el-card class="ai-card" shadow="never">
          <p>{{ summary.summary }}</p>
          <div class="summary-info">
            <el-tag size="small">原文长度：{{ summary.original_length }}字</el-tag>
            <el-tag size="small" type="success">摘要长度：{{ summary.summary_length }}字</el-tag>
          </div>
        </el-card>
      </div>

      <!-- AI分析结果 -->
      <div v-if="analysis" class="ai-result-section">
        <el-divider content-position="left">AI内容分析</el-divider>
        <el-card class="ai-card" shadow="never">
          <pre class="analysis-content">{{ analysis.analysis }}</pre>
        </el-card>
      </div>
    </el-card>

    <!-- 分类统计卡片 -->
    <el-card v-if="stats" class="stats-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><PieChart /></el-icon>
          <span>实时统计</span>
        </div>
      </template>
      <div class="stats-grid">
        <div v-for="(count, category) in stats.stats" :key="category" class="stat-item">
          <div class="stat-number">{{ count }}</div>
          <div class="stat-label">{{ category }}</div>
        </div>
      </div>
      <div class="total-count">
        <el-tag type="info" size="large">总计：{{ stats.total }}条</el-tag>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { classifyText as apiClassifyText, generateSummary as apiGenerateSummary, analyzeContent as apiAnalyzeContent } from '../services/api'

const router = useRouter()

const form = reactive({
  text: ''
})

const loading = ref(false)
const summaryLoading = ref(false)
const analysisLoading = ref(false)
const result = ref(null)
const summary = ref(null)
const analysis = ref(null)
const stats = ref(null)

const exampleTexts = [
  '教育部发布最新通知，要求各地加强中小学生心理健康教育工作，建立完善的心理健康服务体系。通知强调，要配备专业的心理健康教师，定期开展心理健康筛查，及时发现和干预学生心理问题。',
  '苹果公司今日发布了最新的iPhone 15系列手机，采用了全新的A17芯片，性能相比上一代提升了20%。新机型还配备了更先进的摄像系统，支持8K视频录制，预计将于下月正式开售。',
  '央行今日宣布下调存款准备金率0.5个百分点，释放流动性约1万亿元。此举旨在支持实体经济发展，降低企业融资成本。分析师认为，这将有助于稳定经济增长，提振市场信心。'
]

const classifyText = async () => {
  if (!form.text.trim()) {
    ElMessage.warning('请输入新闻内容')
    return
  }

  loading.value = true
  summary.value = null
  analysis.value = null

  try {
    const response = await apiClassifyText(form.text)
    result.value = response
    stats.value = response
    ElMessage.success('分类完成！')
  } catch (error) {
    ElMessage.error('分类失败，请重试')
    console.error(error)
  } finally {
    loading.value = false
  }
}

const generateSummary = async () => {
  summaryLoading.value = true
  try {
    const response = await apiGenerateSummary(form.text)
    summary.value = response
    ElMessage.success('摘要生成完成！')
  } catch (error) {
    ElMessage.error('摘要生成失败')
    console.error(error)
  } finally {
    summaryLoading.value = false
  }
}

const analyzeContent = async () => {
  analysisLoading.value = true
  try {
    const response = await apiAnalyzeContent(form.text)
    analysis.value = response
    ElMessage.success('内容分析完成！')
  } catch (error) {
    ElMessage.error('内容分析失败')
    console.error(error)
  } finally {
    analysisLoading.value = false
  }
}

const clearText = () => {
  form.text = ''
  result.value = null
  summary.value = null
  analysis.value = null
}

const loadExample = () => {
  const randomIndex = Math.floor(Math.random() * exampleTexts.length)
  form.text = exampleTexts[randomIndex]
}

const viewStats = () => {
  router.push('/statistics')
}
</script>

<style scoped>
.classification-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

.main-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 18px;
}

.input-section {
  margin-bottom: 20px;
}

.result-section {
  margin: 20px 0;
}

.quick-actions {
  margin-top: 16px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.ai-result-section {
  margin: 20px 0;
}

.ai-card {
  background: #f8f9fa;
}

.summary-info {
  margin-top: 12px;
  display: flex;
  gap: 8px;
}

.analysis-content {
  white-space: pre-wrap;
  font-family: inherit;
  margin: 0;
  line-height: 1.6;
}

.stats-card {
  margin-top: 20px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.stat-item {
  text-align: center;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-number {
  font-size: 24px;
  font-weight: 600;
  color: #409eff;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #606266;
}

.total-count {
  text-align: center;
}
</style>
