<template>
  <div class="statistics-container">
    <!-- 统计概览 -->
    <el-row :gutter="20" class="overview-cards">
      <el-col :span="6">
        <el-card class="overview-card" shadow="hover">
          <div class="card-content">
            <div class="card-icon total">
              <el-icon><Document /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-number">{{ statsData?.total || 0 }}</div>
              <div class="card-label">总分类数</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="overview-card" shadow="hover">
          <div class="card-content">
            <div class="card-icon categories">
              <el-icon><Grid /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-number">7</div>
              <div class="card-label">支持分类</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="overview-card" shadow="hover">
          <div class="card-content">
            <div class="card-icon popular">
              <el-icon><TrendCharts /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-number">{{ mostPopularCategory }}</div>
              <div class="card-label">热门分类</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="overview-card" shadow="hover">
          <div class="card-content">
            <div class="card-icon accuracy">
              <el-icon><SuccessFilled /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-number">{{ statsData?.accuracy || 0 }}%</div>
              <div class="card-label">准确率</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="20" class="charts-row">
      <!-- 饼图 -->
      <el-col :span="12">
        <el-card class="chart-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><PieChart /></el-icon>
              <span>分类分布饼图</span>
              <el-button type="primary" size="small" @click="refreshData">
                <el-icon><Refresh /></el-icon>
                刷新
              </el-button>
            </div>
          </template>
          <div ref="pieChartRef" class="chart-container"></div>
        </el-card>
      </el-col>

      <!-- 柱状图 -->
      <el-col :span="12">
        <el-card class="chart-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><BarChart /></el-icon>
              <span>分类统计柱状图</span>
            </div>
          </template>
          <div ref="barChartRef" class="chart-container"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 详细数据表格 -->
    <el-card class="table-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><List /></el-icon>
          <span>详细统计数据</span>
          <el-button type="danger" size="small" @click="clearAllData">
            <el-icon><Delete /></el-icon>
            清空数据
          </el-button>
        </div>
      </template>
      <el-table :data="tableData" style="width: 100%">
        <el-table-column prop="category" label="分类" width="120">
          <template #default="scope">
            <el-tag :type="getCategoryTagType(scope.row.category)">
              {{ scope.row.category }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="count" label="数量" width="100" sortable />
        <el-table-column prop="percentage" label="占比" width="120">
          <template #default="scope">
            <el-progress 
              :percentage="scope.row.percentage" 
              :stroke-width="8"
              :show-text="true"
              :format="() => `${scope.row.percentage}%`"
            />
          </template>
        </el-table-column>
        <el-table-column label="趋势" width="100">
          <template #default="scope">
            <el-icon class="trend-icon" :class="getTrendClass(scope.row.percentage)">
              <component :is="getTrendIcon(scope.row.percentage)" />
            </el-icon>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" />
      </el-table>
    </el-card>

    <!-- 准确率详细信息 -->
    <el-card class="accuracy-card" shadow="hover" v-if="accuracyData">
      <template #header>
        <div class="card-header">
          <el-icon><DataAnalysis /></el-icon>
          <span>准确率详细分析</span>
        </div>
      </template>
      <el-row :gutter="20">
        <el-col :span="8">
          <div class="accuracy-summary">
            <div class="summary-item">
              <span class="label">总体准确率：</span>
              <span class="value">{{ accuracyData.overall_accuracy }}%</span>
            </div>
            <div class="summary-item">
              <span class="label">总比较次数：</span>
              <span class="value">{{ accuracyData.total_comparisons }}</span>
            </div>
            <div class="summary-item">
              <span class="label">正确预测：</span>
              <span class="value">{{ accuracyData.correct_predictions }}</span>
            </div>
          </div>
        </el-col>
        <el-col :span="16">
          <div class="category-accuracy">
            <h4>各分类准确率</h4>
            <el-row :gutter="10">
              <el-col :span="8" v-for="(data, category) in accuracyData.category_accuracy" :key="category">
                <div class="category-accuracy-item">
                  <div class="category-name">{{ category }}</div>
                  <el-progress 
                    :percentage="data.accuracy" 
                    :stroke-width="6"
                    :show-text="true"
                    :format="() => `${data.accuracy}%`"
                  />
                  <div class="category-stats">{{ data.correct }}/{{ data.total }}</div>
                </div>
              </el-col>
            </el-row>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { getStats, clearHistory, getAccuracyDetails } from '../services/api'

const pieChartRef = ref<HTMLElement>()
const barChartRef = ref<HTMLElement>()
const statsData = ref(null)
const accuracyData = ref(null)
const loading = ref(false)

let pieChart: echarts.ECharts | null = null
let barChart: echarts.ECharts | null = null

const categoryDescriptions = {
  '教育': '教育相关新闻，包括教育政策、学校动态等',
  '科技': '科技创新、产品发布、技术发展等新闻',
  '社会': '社会民生、公共事务、社会现象等新闻',
  '时政': '政治新闻、政策解读、时事评论等',
  '财经': '经济动态、金融市场、商业资讯等新闻',
  '房产': '房地产市场、政策变化、楼市动态等',
  '家居': '家居装修、生活用品、居家生活等新闻'
}

const mostPopularCategory = computed(() => {
  if (!statsData.value?.stats) return '暂无'
  const stats = statsData.value.stats
  const maxCategory = Object.keys(stats).reduce((a, b) => stats[a] > stats[b] ? a : b)
  return maxCategory
})

const tableData = computed(() => {
  if (!statsData.value) return []
  
  return Object.entries(statsData.value.stats).map(([category, count]) => ({
    category,
    count,
    percentage: statsData.value.percentages[category] || 0,
    description: categoryDescriptions[category] || '暂无描述'
  }))
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

const getTrendClass = (percentage: number) => {
  if (percentage > 20) return 'trend-up'
  if (percentage < 5) return 'trend-down'
  return 'trend-stable'
}

const getTrendIcon = (percentage: number) => {
  if (percentage > 20) return 'CaretTop'
  if (percentage < 5) return 'CaretBottom'
  return 'Minus'
}

const initCharts = () => {
  nextTick(() => {
    if (pieChartRef.value) {
      pieChart = echarts.init(pieChartRef.value)
    }
    if (barChartRef.value) {
      barChart = echarts.init(barChartRef.value)
    }
    updateCharts()
  })
}

const updateCharts = () => {
  if (!statsData.value) return

  const categories = Object.keys(statsData.value.stats)
  const values = Object.values(statsData.value.stats)

  // 饼图配置
  const pieOption = {
    title: {
      text: '新闻分类分布',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 'bold'
      }
    },
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      top: 'middle'
    },
    series: [
      {
        name: '分类统计',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['60%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: false,
          position: 'center'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 20,
            fontWeight: 'bold'
          }
        },
        labelLine: {
          show: false
        },
        data: categories.map((category, index) => ({
          value: values[index],
          name: category,
          itemStyle: {
            color: [
              '#5470c6', '#91cc75', '#fac858', '#ee6666',
              '#73c0de', '#3ba272', '#fc8452'
            ][index % 7]
          }
        }))
      }
    ]
  }

  // 柱状图配置
  const barOption = {
    title: {
      text: '各分类数量统计',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 'bold'
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: categories,
      axisTick: {
        alignWithLabel: true
      }
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '数量',
        type: 'bar',
        barWidth: '60%',
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#83bff6' },
            { offset: 0.5, color: '#188df0' },
            { offset: 1, color: '#188df0' }
          ])
        },
        data: values
      }
    ]
  }

  pieChart?.setOption(pieOption)
  barChart?.setOption(barOption)
}

const refreshData = async () => {
  loading.value = true
  try {
    const [statsResponse, accuracyResponse] = await Promise.all([
      getStats(),
      getAccuracyDetails()
    ])
    statsData.value = statsResponse
    accuracyData.value = accuracyResponse
    updateCharts()
    ElMessage.success('数据刷新成功')
  } catch (error) {
    ElMessage.error('数据加载失败')
    console.error(error)
  } finally {
    loading.value = false
  }
}

const clearAllData = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要清空所有历史数据吗？此操作不可恢复。',
      '警告',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    
    await clearHistory()
    await refreshData()
    ElMessage.success('数据清空成功')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('清空失败')
      console.error(error)
    }
  }
}

onMounted(() => {
  refreshData()
  initCharts()
  
  // 监听窗口大小变化
  window.addEventListener('resize', () => {
    pieChart?.resize()
    barChart?.resize()
  })
})
</script>

<style scoped>
.statistics-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
  border-radius: 8px;
}

.overview-cards {
  margin-bottom: 30px;
  padding: 0 5px;
}

.overview-card {
  height: 120px;
  transition: all 0.3s ease;
  border: 1px solid transparent;
}

.overview-card:hover {
  transform: translateY(-4px);
  border-color: #409eff;
  box-shadow: 0 8px 25px rgba(64, 158, 255, 0.15);
}

.card-content {
  display: flex;
  align-items: center;
  height: 100%;
  padding: 8px 0;
  gap: 12px; /* 添加间距 */
}

.card-icon {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
  flex-shrink: 0; /* 防止图标被压缩 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
}

.card-icon:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

.card-icon.total {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.card-icon.categories {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.card-icon.popular {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.card-icon.accuracy {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.card-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  min-width: 0; /* 防止内容溢出 */
}

.card-number {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
  line-height: 1.2;
}

.card-label {
  font-size: 14px;
  color: #909399;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-align: center;
  margin-top: 4px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.charts-row {
  margin-bottom: 30px;
  padding: 0 5px;
}

.chart-card {
  height: 400px;
  transition: all 0.3s ease;
}

.chart-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.chart-container {
  height: 320px;
}

.table-card {
  margin-bottom: 30px;
  padding: 0 5px;
  transition: all 0.3s ease;
}

.table-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 600;
  font-size: 16px;
}

.card-header span {
  display: flex;
  align-items: center;
  gap: 8px;
}

.trend-icon {
  font-size: 18px;
}

.trend-up {
  color: #67c23a;
}

.trend-down {
  color: #f56c6c;
}

.trend-stable {
  color: #909399;
}

.accuracy-card {
  margin-top: 20px;
}

.accuracy-summary {
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.summary-item {
  margin-bottom: 12px;
}

.label {
  font-weight: 500;
  color: #606266;
}

.value {
  font-weight: 600;
  color: #303133;
}

.category-accuracy {
  background-color: #fff;
  padding: 16px;
  border-radius: 8px;
}

.category-accuracy-item {
  margin-bottom: 16px;
}

.category-name {
  font-weight: 500;
  color: #303133;
  margin-bottom: 8px;
}

.category-stats {
  font-size: 12px;
  color: #909399;
  text-align: center;
  margin-top: 4px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .statistics-container {
    padding: 10px;
  }
  
  .overview-cards .el-col {
    margin-bottom: 15px;
  }
  
  .card-content {
    flex-direction: column;
    text-align: center;
    gap: 8px;
  }
  
  .card-icon {
    width: 50px;
    height: 50px;
    font-size: 20px;
  }
  
  .card-number {
    font-size: 24px;
  }
  
  .card-label {
    font-size: 12px;
  }
  
  .chart-card {
    height: 300px;
  }
  
  .chart-container {
    height: 220px;
  }
}

@media (max-width: 992px) {
  .charts-row .el-col {
    margin-bottom: 20px;
  }
}

/* 提升可访问性 */
.card-content:focus-within {
  outline: 2px solid #409eff;
  outline-offset: 2px;
  border-radius: 4px;
}

/* 动画效果 */
.overview-card {
  animation: fadeInUp 0.6s ease-out;
}

.overview-card:nth-child(1) { animation-delay: 0.1s; }
.overview-card:nth-child(2) { animation-delay: 0.2s; }
.overview-card:nth-child(3) { animation-delay: 0.3s; }
.overview-card:nth-child(4) { animation-delay: 0.4s; }

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 优化文本显示 */
.card-number {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-size: 200% 200%;
  animation: gradientShift 3s ease infinite;
}

@keyframes gradientShift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

/* 卡片内容对齐优化 */
.card-info {
  text-align: left;
}

.card-info .card-number,
.card-info .card-label {
  display: block;
  width: 100%;
}

/* 改进悬停效果 */
.overview-card:hover .card-icon {
  transform: scale(1.1) rotate(5deg);
}

.overview-card:hover .card-number {
  animation-play-state: paused;
}

.overview-card:hover .card-label {
  color: #606266;
  font-weight: 600;
}
</style>
