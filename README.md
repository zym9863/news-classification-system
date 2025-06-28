# 智能新闻分类系统

基于深度学习和AI技术的新闻自动分类系统，集成数据可视化和智能分析功能。

## 🚀 系统特性

### 核心功能
- **🤖 AI智能分类**: 基于Transformer模型的新闻自动分类
- **📊 数据可视化**: 使用ECharts展示分类统计和趋势分析
- **📝 智能摘要**: 集成Pollinations AI的文本摘要生成
- **🔍 内容分析**: AI深度分析文本内容和情感倾向
- **📈 实时统计**: 动态统计分类数据和历史记录

### 支持分类
- 教育 📚
- 科技 💻
- 社会 🏛️
- 时政 🏛️
- 财经 💰
- 房产 🏠
- 家居 🏡

## 🛠️ 技术栈

### 前端
- **Vue 3** + TypeScript
- **Element Plus** - UI组件库
- **ECharts** - 数据可视化
- **Vue Router** - 路由管理
- **Axios** - HTTP客户端

### 后端
- **Flask** - Web框架
- **TensorFlow** - 深度学习模型
- **Pollinations AI** - AI文本生成
- **Pandas** - 数据处理
- **Jieba** - 中文分词

### AI集成
- **Pollinations API** - 无需API Key的AI服务
- **Transformer模型** - 新闻分类
- **自然语言处理** - 文本预处理

## 📦 安装和运行

### 环境要求
- Python 3.6+
- Node.js 16+
- npm 或 pnpm

### 后端安装
```bash
cd backend
pip install flask flask-cors tensorflow pandas jieba requests aiohttp
```

### 前端安装
```bash
cd frontend
npm install
# 或使用 pnpm
pnpm install
```

### 启动系统

#### 方式一：使用启动脚本（Windows）
```bash
start_system.bat
```

#### 方式二：手动启动
```bash
# 启动后端
cd backend
python app.py

# 启动前端（新终端）
cd frontend
pnpm run dev
```

### 访问地址
- 前端界面: http://localhost:5173
- 后端API: http://localhost:5000

## 🎯 功能模块

### 1. 新闻分类
- 输入新闻文本，AI自动识别分类
- 支持批量分类和历史记录
- 实时显示分类统计

### 2. 数据统计
- 饼图展示分类分布
- 柱状图显示数量统计
- 实时更新统计数据
- 支持数据导出

### 3. 历史记录
- 查看所有分类历史
- 支持搜索和筛选
- 重新分类功能
- 批量操作支持

### 4. AI工具
- **文本生成**: 基于提示词生成文本
- **智能摘要**: 自动提取文本关键信息
- **内容分析**: 深度分析文本内容和情感

## 🔧 API接口

### 新闻分类
```http
POST /classify
Content-Type: application/json

{
  "text": "新闻内容"
}
```

### 获取统计数据
```http
GET /stats
```

### AI文本生成
```http
POST /ai/generate
Content-Type: application/json

{
  "prompt": "生成提示词",
  "model": "openai-large"
}
```

### AI摘要生成
```http
POST /ai/summarize
Content-Type: application/json

{
  "text": "需要摘要的文本"
}
```

### AI内容分析
```http
POST /ai/analyze
Content-Type: application/json

{
  "text": "需要分析的文本"
}
```

## 📊 数据可视化

### 统计图表
- **饼图**: 分类分布占比
- **柱状图**: 各分类数量对比
- **趋势图**: 分类趋势变化
- **进度条**: 实时统计进度

### 交互功能
- 图表缩放和平移
- 数据点击查看详情
- 实时数据更新
- 响应式设计

## 🤖 AI功能详解

### Pollinations AI集成
- 无需API Key，免费使用
- 支持多种AI模型
- 私有化处理，保护数据安全
- 高质量文本生成

### 智能分析
- 主要观点提取
- 关键信息识别
- 情感倾向分析
- 重要性评级

## 🎨 界面设计

### 现代化UI
- 响应式设计，支持多设备
- 深色/浅色主题切换
- 流畅的动画效果
- 直观的用户体验

### 组件化架构
- 可复用的Vue组件
- 统一的设计语言
- 模块化的代码结构
- 易于维护和扩展

## 🔒 数据安全

### 隐私保护
- 本地数据存储
- 私有AI处理
- 无用户数据收集
- 安全的API调用

## 📈 性能优化

### 前端优化
- 组件懒加载
- 图片压缩优化
- 代码分割
- 缓存策略

### 后端优化
- 模型预加载
- 异步处理
- 数据缓存
- 错误处理

## 🚀 未来规划

- [ ] 支持更多新闻分类
- [ ] 增加多语言支持
- [ ] 集成更多AI模型
- [ ] 添加用户管理系统
- [ ] 支持批量文件处理
- [ ] 移动端适配优化

## 📝 更新日志

### v1.0.0 (2024-06-28)
- ✅ 完成基础新闻分类功能
- ✅ 集成Pollinations AI
- ✅ 实现数据可视化
- ✅ 添加历史记录管理
- ✅ 开发AI工具套件

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

---

**智能新闻分类系统** - 让AI为新闻分类赋能 🚀
