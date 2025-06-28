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
              <div class="card-number">95%</div>
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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { getStats, clearHistory } from '../services/api'

const pieChartRef = ref<HTMLElement>()
const barChartRef = ref<HTMLElement>()
const statsData = ref(null)
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
    const response = await getStats()
    statsData.value = response
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
}

.overview-cards {
  margin-bottom: 20px;
}

.overview-card {
  height: 120px;
}

.card-content {
  display: flex;
  align-items: center;
  height: 100%;
}

.card-icon {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  font-size: 24px;
  color: white;
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

.card-number {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.card-label {
  font-size: 14px;
  color: #909399;
}

.charts-row {
  margin-bottom: 20px;
}

.chart-card {
  height: 400px;
}

.chart-container {
  height: 320px;
}

.table-card {
  margin-bottom: 20px;
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
</style>
