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
      <el-row :gutter="24">
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/classification')">
            <div class="card-content">
              <div class="card-icon-wrapper">
                <el-icon class="card-icon"><Document /></el-icon>
              </div>
              <div class="card-text">
                <h3 class="card-title">新闻分类</h3>
                <p class="card-description">对新闻文本的自动分类</p>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/statistics')">
            <div class="card-content">
              <div class="card-icon-wrapper">
                <el-icon class="card-icon"><PieChart /></el-icon>
              </div>
              <div class="card-text">
                <h3 class="card-title">数据统计</h3>
                <p class="card-description">可视化展示分类统计数据</p>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/history')">
            <div class="card-content">
              <div class="card-icon-wrapper">
                <el-icon class="card-icon"><Clock /></el-icon>
              </div>
              <div class="card-text">
                <h3 class="card-title">历史记录</h3>
                <p class="card-description">查看所有分类历史记录</p>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="feature-card" shadow="hover" @click="$router.push('/ai-tools')">
            <div class="card-content">
              <div class="card-icon-wrapper">
                <el-icon class="card-icon"><MagicStick /></el-icon>
              </div>
              <div class="card-text">
                <h3 class="card-title">AI工具</h3>
                <p class="card-description">文本生成等AI功能</p>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 系统信息 -->
    <el-card class="info-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon class="header-icon"><InfoFilled /></el-icon>
          <span>系统信息</span>
        </div>
      </template>
      <el-row :gutter="24">
        <el-col :span="8">
          <div class="info-item">
            <div class="info-header">
              <el-icon class="info-icon"><CollectionTag /></el-icon>
              <h4>支持分类</h4>
            </div>
            <div class="category-container">
              <el-tag v-for="category in categories" :key="category" class="category-tag">
                {{ category }}
              </el-tag>
            </div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-item">
            <div class="info-header">
              <el-icon class="info-icon"><Platform /></el-icon>
              <h4>技术栈</h4>
            </div>
            <div class="tech-info">
              <p class="tech-item">
                <span class="tech-label">前端：</span>
                <span class="tech-value">Vue 3 + TypeScript + Element Plus + ECharts</span>
              </p>
              <p class="tech-item">
                <span class="tech-label">后端：</span>
                <span class="tech-value">Flask + TensorFlow + Pollinations AI</span>
              </p>
            </div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-item">
            <div class="info-header">
              <el-icon class="info-icon"><Cpu /></el-icon>
              <h4>AI能力</h4>
            </div>
            <div class="ai-capabilities">
              <div class="capability-item">
                <el-icon class="capability-icon"><Check /></el-icon>
                <span>新闻自动分类</span>
              </div>
              <div class="capability-item">
                <el-icon class="capability-icon"><Check /></el-icon>
                <span>智能文本摘要</span>
              </div>
              <div class="capability-item">
                <el-icon class="capability-icon"><Check /></el-icon>
                <span>内容深度分析</span>
              </div>
              <div class="capability-item">
                <el-icon class="capability-icon"><Check /></el-icon>
                <span>文本生成</span>
              </div>
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { InfoFilled, CollectionTag, Platform, Cpu, Check } from '@element-plus/icons-vue'

const categories = ref(['教育', '科技', '社会', '时政', '财经', '房产', '家居'])
</script>

<style scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .home-container {
    padding: 16px;
  }
  
  .welcome-content {
    flex-direction: column;
    text-align: center;
    padding: 30px 16px;
  }
  
  .welcome-text h1 {
    font-size: 28px;
  }
  
  .subtitle {
    font-size: 16px;
  }
  
  .main-icon {
    font-size: 80px;
    margin-top: 20px;
  }
  
  .features {
    justify-content: center;
  }
}

.welcome-card {
  margin-bottom: 40px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 16px;
  overflow: hidden;
}

.welcome-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 50px 40px;
  min-height: 200px;
}

.welcome-text {
  flex: 1;
  max-width: 60%;
}

.welcome-text h1 {
  font-size: 42px;
  margin-bottom: 20px;
  font-weight: 700;
  line-height: 1.2;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.subtitle {
  font-size: 20px;
  margin-bottom: 32px;
  opacity: 0.95;
  line-height: 1.5;
  font-weight: 400;
}

.features {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.features .el-tag {
  font-size: 14px;
  padding: 8px 16px;
  border: none;
  border-radius: 20px;
  font-weight: 500;
}

.welcome-image {
  flex-shrink: 0;
  margin-left: 40px;
}

.main-icon {
  font-size: 140px;
  opacity: 0.9;
  filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
}

.feature-cards {
  margin-bottom: 40px;
}

.feature-card {
  cursor: pointer;
  transition: all 0.3s ease;
  height: 220px;
  border-radius: 12px;
  border: 1px solid #f0f0f0;
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 24px 16px;
}

.card-icon-wrapper {
  margin-bottom: 20px;
}

.card-icon {
  font-size: 52px;
  color: #409eff;
  transition: color 0.3s ease;
}

.feature-card:hover .card-icon {
  color: #337ecc;
}

.card-text {
  text-align: center;
  width: 100%;
}

.card-title {
  margin: 0 0 12px 0;
  color: #303133;
  font-size: 18px;
  font-weight: 600;
  line-height: 1.4;
}

.card-description {
  color: #606266;
  font-size: 14px;
  line-height: 1.6;
  margin: 0;
}

.info-card {
  margin-bottom: 20px;
  border-radius: 12px;
}

.card-header {
  display: flex;
  align-items: center;
  font-weight: 600;
  font-size: 18px;
  color: #303133;
}

.header-icon {
  margin-right: 8px;
  color: #409eff;
}

.info-item {
  padding: 16px;
  height: 100%;
}

.info-header {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
}

.info-icon {
  margin-right: 8px;
  color: #409eff;
  font-size: 18px;
}

.info-item h4 {
  margin: 0;
  color: #303133;
  font-size: 16px;
  font-weight: 600;
}

.category-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.category-tag {
  margin: 0;
  border-radius: 16px;
  font-size: 12px;
  padding: 4px 12px;
}

.tech-item {
  margin: 12px 0;
  line-height: 1.6;
}

.tech-label {
  color: #909399;
  font-weight: 500;
  font-size: 14px;
}

.tech-value {
  color: #606266;
  font-size: 14px;
}

.ai-capabilities {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.capability-item {
  display: flex;
  align-items: center;
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
}

.capability-icon {
  margin-right: 8px;
  color: #67c23a;
  font-size: 16px;
}
</style>
