# 新闻分类系统

基于机器学习的中文新闻智能分类系统，支持7个类别的新闻自动分类：教育、科技、社会、时政、财经、房产、家居。

## 🎯 功能特性

- **智能分类**：基于朴素贝叶斯算法的中文文本分类
- **实时预测**：输入新闻文本即可获得分类结果和置信度
- **现代界面**：基于React + Ant Design的响应式Web界面
- **高性能API**：FastAPI后端提供快速的REST API接口
- **容器化部署**：支持Docker单容器部署
- **中文优化**：使用jieba分词，专门针对中文文本优化

## 🏗️ 技术栈

### 后端
- **FastAPI** - 现代高性能Web框架
- **scikit-learn** - 机器学习库（朴素贝叶斯分类器）
- **jieba** - 中文分词工具
- **pandas** - 数据处理
- **uv** - Python包管理器

### 前端
- **React 19** - 用户界面库
- **TypeScript** - 类型安全的JavaScript
- **Ant Design** - 企业级UI组件库
- **Vite** - 现代前端构建工具
- **Axios** - HTTP客户端
- **pnpm** - 高效的包管理器

### 部署
- **Docker** - 容器化部署
- **多阶段构建** - 优化镜像大小

## 📊 数据集

- **总量**：70,000条中文新闻数据
- **类别**：7个类别，每类10,000条数据
- **格式**：Excel文件（标题 + 类别）
- **分布**：数据均衡，适合分类模型训练

| 类别 | 数量 | 描述 |
|------|------|------|
| 教育 | 10,000 | 教育相关新闻 |
| 科技 | 10,000 | 科技创新新闻 |
| 社会 | 10,000 | 社会民生新闻 |
| 时政 | 10,000 | 政治时事新闻 |
| 财经 | 10,000 | 财经商业新闻 |
| 房产 | 10,000 | 房地产相关新闻 |
| 家居 | 10,000 | 家居装修新闻 |

## 🚀 快速开始

### 方法1：使用启动脚本（推荐）

**Windows用户：**
```bash
# 双击运行或在命令行执行
start.bat
```

**Linux/macOS用户：**
```bash
# 给脚本执行权限并运行
chmod +x start.sh
./start.sh
```

### 方法2：Docker部署

```bash
# 构建并运行
docker-compose up --build

# 后台运行
docker-compose up -d --build
```

### 方法3：手动启动

**后端启动：**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install uv
uv pip install -e .
python main.py
```

**前端启动：**
```bash
cd frontend
pnpm install
pnpm run dev
```

## 🌐 访问地址

- **前端界面**：http://localhost:5173
- **后端API**：http://localhost:8000
- **API文档**：http://localhost:8000/docs
- **Redoc文档**：http://localhost:8000/redoc

## 📖 API接口

### 获取类别列表
```http
GET /api/categories
```

### 单条新闻分类
```http
POST /api/predict
Content-Type: application/json

{
    "text": "教育部发布新的课程标准"
}
```

### 批量新闻分类
```http
POST /api/batch_predict
Content-Type: application/json

{
    "texts": ["新闻1", "新闻2", "新闻3"]
}
```

### 获取模型信息
```http
GET /api/model_info
```

## 🧠 模型详情

- **算法**：多项式朴素贝叶斯（MultinomialNB）
- **特征提取**：TF-IDF向量化
- **分词**：jieba中文分词
- **特征数量**：10,000维
- **n-gram范围**：1-2（单词和双词组合）
- **训练/测试比例**：80%/20%

## 🎨 界面预览

系统提供现代化的Web界面，包含：

- **实时分类**：输入新闻文本，实时获得分类结果
- **置信度显示**：显示预测的置信度百分比
- **历史记录**：保存分类历史，方便对比
- **响应式设计**：支持桌面和移动设备
- **类别标签**：彩色标签显示不同新闻类别

## 📁 项目结构

```
news-classification-system/
├── backend/                 # 后端代码
│   ├── models/             # 机器学习模型
│   │   ├── __init__.py
│   │   └── classifier.py   # 分类器实现
│   ├── data.xlsx          # 训练数据
│   ├── main.py            # FastAPI应用
│   └── pyproject.toml     # Python依赖
├── frontend/               # 前端代码
│   ├── src/
│   │   ├── services/      # API服务
│   │   ├── types/         # TypeScript类型
│   │   ├── App.tsx        # 主组件
│   │   └── main.tsx       # 入口文件
│   └── package.json       # 前端依赖
├── Dockerfile             # Docker镜像配置
├── docker-compose.yml     # Docker编排配置
├── start.sh              # Linux启动脚本
├── start.bat             # Windows启动脚本
└── README.md             # 项目文档
```

## 🔧 开发指南

### 后端开发

1. **环境配置**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install uv
   uv pip install -e .
   ```

2. **模型训练**
   ```bash
   python -m models.classifier
   ```

3. **启动开发服务器**
   ```bash
   python main.py
   ```

### 前端开发

1. **安装依赖**
   ```bash
   cd frontend
   pnpm install
   ```

2. **启动开发服务器**
   ```bash
   pnpm run dev
   ```

3. **构建生产版本**
   ```bash
   pnpm run build
   ```

## 🐳 Docker配置

### Dockerfile说明
- **多阶段构建**：分离前端构建和后端运行环境
- **体积优化**：使用Alpine Linux基础镜像
- **健康检查**：内置服务健康检查
- **安全性**：非root用户运行

### 环境变量
- `PYTHONPATH`: Python模块路径
- `PYTHONUNBUFFERED`: 禁用Python输出缓冲

## 🧪 测试

```bash
# 后端测试
cd backend
python -m pytest

# 前端测试
cd frontend
pnpm run test

# 端到端测试
pnpm run test:e2e
```

## 📈 性能优化

- **模型缓存**：训练好的模型持久化存储
- **异步处理**：FastAPI异步请求处理
- **前端优化**：React.memo和useMemo优化渲染
- **打包优化**：Vite构建优化和代码分割

## 🔒 安全性

- **CORS配置**：跨域请求安全控制
- **输入验证**：API请求参数验证
- **容器安全**：最小权限容器运行

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 确保data.xlsx文件存在
   - 检查Python依赖是否完整安装

2. **前端无法连接后端**
   - 检查后端服务是否启动（端口8000）
   - 确认CORS配置正确

3. **Docker构建失败**
   - 确保Docker已安装并运行
   - 检查磁盘空间是否足够

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

该项目基于MIT许可证开源。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue：[GitHub Issues](https://github.com/your-username/news-classification-system/issues)

---

## 🙏 致谢

感谢以下开源项目和工具：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [React](https://reactjs.org/) - 用户界面库
- [Ant Design](https://ant.design/) - 企业级UI设计语言
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [jieba](https://github.com/fxsjy/jieba) - 中文分词组件