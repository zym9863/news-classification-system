<template>
  <div class="home-container">
    <!-- 欢迎横幅 -->
    <el-card class="welcome-card" shadow="hover">
      <div class="welcome-content">
        <div class="welcome-text">
          <h1>欢迎使用智能新闻分类系统</h1>
          <p class="subtitle">基于深度学习的新闻自动分类，集成AI智能分析与数据可视化</p>
          <div class="features">
            <el-tag type="primary" size="large">🤖 AI智能分类</el-tag>
            <el-tag type="success" size="large">📊 数据可视化</el-tag>
            <el-tag type="warning" size="large">📝 智能摘要</el-tag>
            <el-tag type="info" size="large">📈 趋势分析</el-tag>
          </div>
        </div>
        <div class="welcome-image">
          <el-icon class="main-icon"><TrendCharts /></el-icon>
        </div>
      </div>
    </el-card>

    <!-- 功能卡片 -->
    <div class="feature-cards">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/classification')">
            <div class="card-content">
              <el-icon class="card-icon"><Document /></el-icon>
              <h3>新闻分类</h3>
              <p>输入新闻文本，AI自动识别分类</p>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/statistics')">
            <div class="card-content">
              <el-icon class="card-icon"><PieChart /></el-icon>
              <h3>数据统计</h3>
              <p>可视化展示分类统计数据</p>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/history')">
            <div class="card-content">
              <el-icon class="card-icon"><Clock /></el-icon>
              <h3>历史记录</h3>
              <p>查看所有分类历史记录</p>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/ai-tools')">
            <div class="card-content">
              <el-icon class="card-icon"><MagicStick /></el-icon>
              <h3>AI工具</h3>
              <p>智能摘要、内容分析等AI功能</p>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 系统信息 -->
    <el-card class="info-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>系统信息</span>
        </div>
      </template>
      <el-row :gutter="20">
        <el-col :span="8">
          <div class="info-item">
            <h4>支持分类</h4>
            <el-tag v-for="category in categories" :key="category" class="category-tag">
              {{ category }}
            </el-tag>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-item">
            <h4>技术栈</h4>
            <p>前端：Vue 3 + TypeScript + Element Plus + ECharts</p>
            <p>后端：Flask + TensorFlow + Pollinations AI</p>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-item">
            <h4>AI能力</h4>
            <p>✅ 新闻自动分类</p>
            <p>✅ 智能文本摘要</p>
            <p>✅ 内容深度分析</p>
            <p>✅ 文本生成</p>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const categories = ref(['教育', '科技', '社会', '时政', '财经', '房产', '家居'])
</script>

<style scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.welcome-card {
  margin-bottom: 30px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.welcome-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 40px 20px;
}

.welcome-text h1 {
  font-size: 36px;
  margin-bottom: 16px;
  font-weight: 600;
}

.subtitle {
  font-size: 18px;
  margin-bottom: 24px;
  opacity: 0.9;
}

.features {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.main-icon {
  font-size: 120px;
  opacity: 0.8;
}

.feature-cards {
  margin-bottom: 30px;
}

.feature-card {
  cursor: pointer;
  transition: transform 0.3s ease;
  height: 200px;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.card-content {
  text-align: center;
  padding: 20px;
}

.card-icon {
  font-size: 48px;
  color: #409eff;
  margin-bottom: 16px;
}

.card-content h3 {
  margin-bottom: 12px;
  color: #303133;
}

.card-content p {
  color: #606266;
  font-size: 14px;
}

.info-card {
  margin-bottom: 20px;
}

.card-header {
  font-weight: 600;
  font-size: 18px;
}

.info-item h4 {
  margin-bottom: 12px;
  color: #303133;
}

.category-tag {
  margin: 4px;
}

.info-item p {
  margin: 8px 0;
  color: #606266;
}
</style>
