<template>
  <div class="history-container">
    <el-card class="main-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <el-icon><Clock /></el-icon>
            <span>分类历史记录</span>
          </div>
          <div class="header-right">
            <el-button type="primary" @click="refreshData" :loading="loading">
              <el-icon><Refresh /></el-icon>
              刷新
            </el-button>
            <el-button type="danger" @click="clearAllHistory">
              <el-icon><Delete /></el-icon>
              清空历史
            </el-button>
          </div>
        </div>
      </template>

      <!-- 搜索和筛选 -->
      <div class="filter-section">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-input
              v-model="searchText"
              placeholder="搜索文本内容..."
              clearable
              @input="handleSearch"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
          </el-col>
          <el-col :span="6">
            <el-select
              v-model="selectedCategory"
              placeholder="选择分类"
              clearable
              @change="handleCategoryFilter"
            >
              <el-option
                v-for="category in categories"
                :key="category"
                :label="category"
                :value="category"
              />
            </el-select>
          </el-col>
          <el-col :span="6">
            <el-date-picker
              v-model="dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
              @change="handleDateFilter"
            />
          </el-col>
          <el-col :span="4">
            <el-button type="info" @click="resetFilters">
              <el-icon><RefreshLeft /></el-icon>
              重置筛选
            </el-button>
          </el-col>
        </el-row>
      </div>

      <!-- 历史记录表格 -->
      <el-table
        :data="filteredHistory"
        v-loading="loading"
        style="width: 100%"
        :default-sort="{ prop: 'timestamp', order: 'descending' }"
      >
        <el-table-column prop="id" label="ID" width="80" sortable />
        <el-table-column prop="category" label="分类" width="100">
          <template #default="scope">
            <el-tag :type="getCategoryTagType(scope.row.category)">
              {{ scope.row.category }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="text" label="文本内容" min-width="300">
          <template #default="scope">
            <div class="text-content">
              <p class="text-preview">{{ getTextPreview(scope.row.text) }}</p>
              <el-button
                v-if="scope.row.text.length > 100"
                type="text"
                size="small"
                @click="showFullText(scope.row)"
              >
                查看全文
              </el-button>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="text_length" label="字数" width="80" sortable />
        <el-table-column prop="timestamp" label="分类时间" width="180" sortable>
          <template #default="scope">
            {{ formatTime(scope.row.timestamp) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="scope">
            <el-button type="primary" size="small" @click="reclassify(scope.row)">
              <el-icon><Refresh /></el-icon>
              重新分类
            </el-button>
            <el-button type="success" size="small" @click="generateSummary(scope.row)">
              <el-icon><Document /></el-icon>
              生成摘要
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-section">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="totalRecords"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 全文显示对话框 -->
    <el-dialog
      v-model="fullTextDialogVisible"
      title="完整文本内容"
      width="60%"
      :before-close="handleCloseFullText"
    >
      <div class="full-text-content">
        <el-tag :type="getCategoryTagType(selectedRecord?.category)" class="category-tag">
          {{ selectedRecord?.category }}
        </el-tag>
        <p class="full-text">{{ selectedRecord?.text }}</p>
        <div class="text-info">
          <el-tag size="small">字数：{{ selectedRecord?.text_length }}</el-tag>
          <el-tag size="small" type="info">时间：{{ formatTime(selectedRecord?.timestamp) }}</el-tag>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="fullTextDialogVisible = false">关闭</el-button>
          <el-button type="primary" @click="reclassify(selectedRecord)">重新分类</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 摘要显示对话框 -->
    <el-dialog
      v-model="summaryDialogVisible"
      title="AI智能摘要"
      width="50%"
    >
      <div v-loading="summaryLoading" class="summary-content">
        <div v-if="currentSummary">
          <h4>原文摘要：</h4>
          <p class="summary-text">{{ currentSummary.summary }}</p>
          <div class="summary-stats">
            <el-tag size="small">原文：{{ currentSummary.original_length }}字</el-tag>
            <el-tag size="small" type="success">摘要：{{ currentSummary.summary_length }}字</el-tag>
            <el-tag size="small" type="info">压缩比：{{ compressionRatio }}%</el-tag>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="summaryDialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getHistory, clearHistory, classifyText, generateSummary as apiGenerateSummary } from '../services/api'

const router = useRouter()

const loading = ref(false)
const summaryLoading = ref(false)
const historyData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const totalRecords = ref(0)

// 筛选相关
const searchText = ref('')
const selectedCategory = ref('')
const dateRange = ref([])

// 对话框相关
const fullTextDialogVisible = ref(false)
const summaryDialogVisible = ref(false)
const selectedRecord = ref(null)
const currentSummary = ref(null)

const categories = ['教育', '科技', '社会', '时政', '财经', '房产', '家居']

const filteredHistory = computed(() => {
  let filtered = [...historyData.value]

  // 文本搜索
  if (searchText.value) {
    filtered = filtered.filter(item =>
      item.text.toLowerCase().includes(searchText.value.toLowerCase())
    )
  }

  // 分类筛选
  if (selectedCategory.value) {
    filtered = filtered.filter(item => item.category === selectedCategory.value)
  }

  // 日期筛选
  if (dateRange.value && dateRange.value.length === 2) {
    const [startDate, endDate] = dateRange.value
    filtered = filtered.filter(item => {
      const itemDate = item.timestamp.split('T')[0]
      return itemDate >= startDate && itemDate <= endDate
    })
  }

  return filtered
})

const compressionRatio = computed(() => {
  if (!currentSummary.value) return 0
  const ratio = (currentSummary.value.summary_length / currentSummary.value.original_length) * 100
  return Math.round(ratio)
})

const getCategoryTagType = (category: string) => {
  const types = {
    '教育': 'primary',
    '科技': 'success',
    '社会': 'info',
    '时政': 'warning',
    '财经': 'danger',
    '房产': '',
    '家居': 'success'
  }
  return types[category] || ''
}

const getTextPreview = (text: string) => {
  return text.length > 100 ? text.substring(0, 100) + '...' : text
}

const formatTime = (timestamp: string) => {
  return new Date(timestamp).toLocaleString('zh-CN')
}

const refreshData = async () => {
  loading.value = true
  try {
    const response = await getHistory(currentPage.value, pageSize.value)
    historyData.value = response.history
    totalRecords.value = response.total
  } catch (error) {
    ElMessage.error('数据加载失败')
    console.error(error)
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  // 搜索逻辑已在computed中处理
}

const handleCategoryFilter = () => {
  // 筛选逻辑已在computed中处理
}

const handleDateFilter = () => {
  // 日期筛选逻辑已在computed中处理
}

const resetFilters = () => {
  searchText.value = ''
  selectedCategory.value = ''
  dateRange.value = []
}

const handleSizeChange = (val: number) => {
  pageSize.value = val
  refreshData()
}

const handleCurrentChange = (val: number) => {
  currentPage.value = val
  refreshData()
}

const showFullText = (record: any) => {
  selectedRecord.value = record
  fullTextDialogVisible.value = true
}

const handleCloseFullText = () => {
  fullTextDialogVisible.value = false
  selectedRecord.value = null
}

const reclassify = async (record: any) => {
  try {
    const response = await classifyText(record.text)
    ElMessage.success(`重新分类完成：${response.category}`)
    refreshData()
  } catch (error) {
    ElMessage.error('重新分类失败')
    console.error(error)
  }
}

const generateSummary = async (record: any) => {
  summaryLoading.value = true
  summaryDialogVisible.value = true
  currentSummary.value = null

  try {
    const response = await apiGenerateSummary(record.text)
    currentSummary.value = response
  } catch (error) {
    ElMessage.error('摘要生成失败')
    console.error(error)
    summaryDialogVisible.value = false
  } finally {
    summaryLoading.value = false
  }
}

const clearAllHistory = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要清空所有历史记录吗？此操作不可恢复。',
      '警告',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    
    await clearHistory()
    await refreshData()
    ElMessage.success('历史记录清空成功')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('清空失败')
      console.error(error)
    }
  }
}

onMounted(() => {
  refreshData()
})
</script>

<style scoped>
.history-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 18px;
}

.header-right {
  display: flex;
  gap: 12px;
}

.filter-section {
  margin-bottom: 20px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.text-content {
  max-width: 300px;
}

.text-preview {
  margin: 0 0 8px 0;
  line-height: 1.5;
  word-break: break-word;
}

.pagination-section {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.full-text-content {
  max-height: 400px;
  overflow-y: auto;
}

.category-tag {
  margin-bottom: 16px;
}

.full-text {
  line-height: 1.8;
  margin: 16px 0;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  word-break: break-word;
}

.text-info {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.summary-content {
  min-height: 200px;
}

.summary-text {
  line-height: 1.8;
  margin: 16px 0;
  padding: 16px;
  background: #f0f9ff;
  border-radius: 8px;
  border-left: 4px solid #409eff;
}

.summary-stats {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}
</style>
