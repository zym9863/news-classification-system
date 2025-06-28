<template>
  <div class="ai-tools-container">
    <!-- AI工具导航 -->
    <el-card class="nav-card" shadow="hover">
      <el-tabs v-model="activeTab" type="card" @tab-click="handleTabClick">
        <el-tab-pane label="文本生成" name="generate">
          <template #label>
            <span class="tab-label">
              <el-icon><Edit /></el-icon>
              文本生成
            </span>
          </template>
        </el-tab-pane>
        <el-tab-pane label="智能摘要" name="summarize">
          <template #label>
            <span class="tab-label">
              <el-icon><Document /></el-icon>
              智能摘要
            </span>
          </template>
        </el-tab-pane>
        <el-tab-pane label="内容分析" name="analyze">
          <template #label>
            <span class="tab-label">
              <el-icon><Search /></el-icon>
              内容分析
            </span>
          </template>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 文本生成工具 -->
    <el-card v-show="activeTab === 'generate'" class="tool-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><MagicStick /></el-icon>
          <span>AI文本生成</span>
          <el-tag type="info">基于Pollinations AI</el-tag>
        </div>
      </template>
      
      <div class="tool-content">
        <el-form :model="generateForm" label-width="100px">
          <el-form-item label="生成主题">
            <el-input
              v-model="generateForm.prompt"
              type="textarea"
              :rows="4"
              placeholder="请输入您想要生成的文本主题或提示词..."
              maxlength="500"
              show-word-limit
            />
          </el-form-item>
          <el-form-item label="AI模型">
            <el-select v-model="generateForm.model" placeholder="选择AI模型">
              <el-option label="GPT 4.1" value="openai-large" />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button 
              type="primary" 
              @click="generateText"
              :loading="generateLoading"
              :disabled="!generateForm.prompt.trim()"
              size="large"
            >
              <el-icon><MagicStick /></el-icon>
              生成文本
            </el-button>
            <el-button @click="clearGenerate" size="large">
              <el-icon><Delete /></el-icon>
              清空
            </el-button>
          </el-form-item>
        </el-form>

        <!-- 生成结果 -->
        <div v-if="generateResult" class="result-section">
          <el-divider content-position="left">生成结果</el-divider>
          <el-card class="result-card" shadow="never">
            <div class="result-header">
              <el-tag type="success">生成完成</el-tag>
              <el-button type="text" @click="copyToClipboard(generateResult.result)">
                <el-icon><CopyDocument /></el-icon>
                复制
              </el-button>
            </div>
            <div class="result-content">
              <pre class="generated-text">{{ generateResult.result }}</pre>
            </div>
            <div class="result-footer">
              <el-tag size="small">模型：{{ generateResult.model }}</el-tag>
              <el-tag size="small" type="info">字数：{{ generateResult.result.length }}</el-tag>
            </div>
          </el-card>
        </div>
      </div>
    </el-card>

    <!-- 智能摘要工具 -->
    <el-card v-show="activeTab === 'summarize'" class="tool-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Document /></el-icon>
          <span>AI智能摘要</span>
          <el-tag type="success">自动提取关键信息</el-tag>
        </div>
      </template>
      
      <div class="tool-content">
        <el-form :model="summaryForm" label-width="100px">
          <el-form-item label="原文内容">
            <el-input
              v-model="summaryForm.text"
              type="textarea"
              :rows="8"
              placeholder="请输入需要生成摘要的文本内容..."
              maxlength="5000"
              show-word-limit
            />
          </el-form-item>
          <el-form-item>
            <el-button 
              type="primary" 
              @click="generateSummary"
              :loading="summaryLoading"
              :disabled="!summaryForm.text.trim()"
              size="large"
            >
              <el-icon><Document /></el-icon>
              生成摘要
            </el-button>
            <el-button @click="clearSummary" size="large">
              <el-icon><Delete /></el-icon>
              清空
            </el-button>
            <el-button @click="loadSampleText" size="large">
              <el-icon><Reading /></el-icon>
              示例文本
            </el-button>
          </el-form-item>
        </el-form>

        <!-- 摘要结果 -->
        <div v-if="summaryResult" class="result-section">
          <el-divider content-position="left">摘要结果</el-divider>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-card class="result-card" shadow="never">
                <template #header>
                  <span>原文内容</span>
                </template>
                <div class="original-text">{{ summaryForm.text }}</div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card class="result-card" shadow="never">
                <template #header>
                  <span>智能摘要</span>
                  <el-button type="text" @click="copyToClipboard(summaryResult.summary)">
                    <el-icon><CopyDocument /></el-icon>
                    复制
                  </el-button>
                </template>
                <div class="summary-text">{{ summaryResult.summary }}</div>
                <div class="summary-stats">
                  <el-progress 
                    :percentage="compressionRatio" 
                    :stroke-width="8"
                    :format="() => `压缩率 ${compressionRatio}%`"
                  />
                  <div class="stats-tags">
                    <el-tag size="small">原文：{{ summaryResult.original_length }}字</el-tag>
                    <el-tag size="small" type="success">摘要：{{ summaryResult.summary_length }}字</el-tag>
                  </div>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>
      </div>
    </el-card>

    <!-- 内容分析工具 -->
    <el-card v-show="activeTab === 'analyze'" class="tool-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><TrendCharts /></el-icon>
          <span>AI内容分析</span>
          <el-tag type="warning">深度解读文本</el-tag>
        </div>
      </template>
      
      <div class="tool-content">
        <el-form :model="analysisForm" label-width="100px">
          <el-form-item label="分析内容">
            <el-input
              v-model="analysisForm.text"
              type="textarea"
              :rows="8"
              placeholder="请输入需要分析的文本内容..."
              maxlength="3000"
              show-word-limit
            />
          </el-form-item>
          <el-form-item>
            <el-button 
              type="primary" 
              @click="analyzeContent"
              :loading="analysisLoading"
              :disabled="!analysisForm.text.trim()"
              size="large"
            >
              <el-icon><Search /></el-icon>
              开始分析
            </el-button>
            <el-button @click="clearAnalysis" size="large">
              <el-icon><Delete /></el-icon>
              清空
            </el-button>
          </el-form-item>
        </el-form>

        <!-- 分析结果 -->
        <div v-if="analysisResult" class="result-section">
          <el-divider content-position="left">分析结果</el-divider>
          <el-card class="result-card analysis-card" shadow="never">
            <div class="analysis-header">
              <el-tag type="warning">AI深度分析</el-tag>
              <el-button type="text" @click="copyToClipboard(analysisResult.analysis)">
                <el-icon><CopyDocument /></el-icon>
                复制分析
              </el-button>
            </div>
            <div class="analysis-content">
              <pre class="analysis-text">{{ analysisResult.analysis }}</pre>
            </div>
            <div class="analysis-footer">
              <el-tag size="small">分析字数：{{ analysisResult.text_length }}</el-tag>
              <el-tag size="small" type="info">分析时间：{{ new Date().toLocaleString() }}</el-tag>
            </div>
          </el-card>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { generateAIText, generateSummary as apiGenerateSummary, analyzeContent as apiAnalyzeContent } from '../services/api'

const activeTab = ref('generate')

// 文本生成相关
const generateForm = reactive({
  prompt: '',
  model: 'openai-large'
})
const generateLoading = ref(false)
const generateResult = ref(null)

// 智能摘要相关
const summaryForm = reactive({
  text: ''
})
const summaryLoading = ref(false)
const summaryResult = ref(null)

// 内容分析相关
const analysisForm = reactive({
  text: ''
})
const analysisLoading = ref(false)
const analysisResult = ref(null)

const compressionRatio = computed(() => {
  if (!summaryResult.value) return 0
  const ratio = (summaryResult.value.summary_length / summaryResult.value.original_length) * 100
  return Math.round(100 - ratio)
})

const sampleTexts = [
  '人工智能技术在近年来取得了突破性进展，特别是在自然语言处理、计算机视觉和机器学习等领域。深度学习算法的发展使得AI系统能够处理更复杂的任务，从语音识别到图像分析，再到自动驾驶汽车。然而，随着AI技术的快速发展，也带来了一些挑战，包括数据隐私、算法偏见和就业影响等问题。专家们认为，需要建立完善的AI治理框架，确保技术发展与社会责任并重。',
  '全球气候变化问题日益严重，各国政府和国际组织正在采取积极措施应对这一挑战。联合国气候变化大会强调了减少温室气体排放的紧迫性，提出了到2050年实现碳中和的目标。可再生能源技术的快速发展为实现这一目标提供了可能，太阳能和风能成本大幅下降，电动汽车普及率不断提高。同时，各国也在推进绿色金融，支持清洁技术创新和可持续发展项目。'
]

const handleTabClick = (tab: any) => {
  activeTab.value = tab.name
}

const generateText = async () => {
  generateLoading.value = true
  try {
    const response = await generateAIText(generateForm.prompt, generateForm.model)
    generateResult.value = response
    ElMessage.success('文本生成完成！')
  } catch (error) {
    ElMessage.error('文本生成失败')
    console.error(error)
  } finally {
    generateLoading.value = false
  }
}

const generateSummary = async () => {
  summaryLoading.value = true
  try {
    const response = await apiGenerateSummary(summaryForm.text)
    summaryResult.value = response
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
    const response = await apiAnalyzeContent(analysisForm.text)
    analysisResult.value = response
    ElMessage.success('内容分析完成！')
  } catch (error) {
    ElMessage.error('内容分析失败')
    console.error(error)
  } finally {
    analysisLoading.value = false
  }
}

const clearGenerate = () => {
  generateForm.prompt = ''
  generateResult.value = null
}

const clearSummary = () => {
  summaryForm.text = ''
  summaryResult.value = null
}

const clearAnalysis = () => {
  analysisForm.text = ''
  analysisResult.value = null
}

const loadSampleText = () => {
  const randomIndex = Math.floor(Math.random() * sampleTexts.length)
  summaryForm.text = sampleTexts[randomIndex]
}

const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success('已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败')
  }
}
</script>

<style scoped>
.ai-tools-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.nav-card {
  margin-bottom: 20px;
}

.tab-label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tool-card {
  min-height: 600px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 600;
  font-size: 18px;
}

.tool-content {
  padding: 20px 0;
}

.result-section {
  margin-top: 30px;
}

.result-card {
  margin-bottom: 20px;
  background: #f8f9fa;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.result-content {
  margin: 16px 0;
}

.generated-text,
.analysis-text {
  white-space: pre-wrap;
  font-family: inherit;
  margin: 0;
  line-height: 1.6;
  background: white;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #e4e7ed;
}

.result-footer,
.analysis-footer {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.original-text,
.summary-text {
  line-height: 1.6;
  margin-bottom: 16px;
  max-height: 200px;
  overflow-y: auto;
  padding: 12px;
  background: white;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.summary-stats {
  margin-top: 16px;
}

.stats-tags {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.analysis-card {
  background: #fff7e6;
  border: 1px solid #ffd591;
}

.analysis-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.analysis-content {
  margin: 16px 0;
}
</style>
