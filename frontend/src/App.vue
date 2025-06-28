<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const activeIndex = ref('/')

const handleSelect = (key: string) => {
  activeIndex.value = key
  router.push(key)
}
</script>

<template>
  <div class="app-container">
    <!-- 顶部导航栏 -->
    <el-header class="header">
      <div class="header-content">
        <div class="logo-section">
          <el-icon class="logo-icon"><Document /></el-icon>
          <h1 class="title">智能新闻分类系统</h1>
        </div>
        <el-menu
          :default-active="activeIndex"
          class="nav-menu"
          mode="horizontal"
          @select="handleSelect"
        >
          <el-menu-item index="/">首页</el-menu-item>
          <el-menu-item index="/classification">新闻分类</el-menu-item>
          <el-menu-item index="/statistics">数据统计</el-menu-item>
          <el-menu-item index="/history">历史记录</el-menu-item>
          <el-menu-item index="/ai-tools">AI工具</el-menu-item>
        </el-menu>
      </div>
    </el-header>

    <!-- 主要内容区域 -->
    <el-main class="main-content">
      <router-view />
    </el-main>

    <!-- 底部 -->
    <el-footer class="footer">
      <p>&copy; 2025 智能新闻分类系统 - 基于Flask + Vue + ECharts + AI</p>
    </el-footer>
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0;
  height: 80px;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 32px;
  color: white;
}

.title {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.nav-menu {
  background: transparent;
  border: none;
}

.nav-menu .el-menu-item {
  color: white;
  border-bottom: 2px solid transparent;
}

.nav-menu .el-menu-item:hover,
.nav-menu .el-menu-item.is-active {
  background: rgba(255, 255, 255, 0.1);
  border-bottom-color: white;
  color: white;
}

.main-content {
  flex: 1;
  background: #f5f7fa;
  padding: 20px;
}

.footer {
  background: #2c3e50;
  color: white;
  text-align: center;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.footer p {
  margin: 0;
}
</style>
