[English](README-EN.md) | [简体中文](README.md)

# News Classification System

An ML-powered Chinese news classification system that automatically classifies news into 7 categories: Education, Technology, Society, Current Affairs, Finance, Real Estate, and Home & Living.

## 🎯 Features

- Intelligent classification with a Naive Bayes algorithm optimized for Chinese text
- Real-time prediction: input news text to get category and confidence
- Modern UI: Responsive web interface built with React + Ant Design
- High-performance API: FastAPI backend with fast REST endpoints
- Containerized deployment: Single-container Docker support
- Chinese text optimization: jieba-based tokenization

## 🏗️ Tech Stack

### Backend
- FastAPI – modern, high-performance web framework
- scikit-learn – machine learning (Multinomial Naive Bayes)
- jieba – Chinese word segmentation
- pandas – data processing
- uv – Python package manager

### Frontend
- React 19 – UI library
- TypeScript – type-safe JavaScript
- Ant Design – enterprise-grade UI components
- Vite – modern frontend tooling
- Axios – HTTP client
- pnpm – efficient package manager

### Deployment
- Docker – containerized deployment
- Multi-stage build – optimized image size

## 📊 Dataset

- Total: 70,000 Chinese news records
- Categories: 7 categories, 10,000 samples each
- Format: Excel (title + category)
- Distribution: balanced for classification

| Category | Count | Description |
|---------|-------|-------------|
| Education (教育) | 10,000 | Education-related news |
| Technology (科技) | 10,000 | Tech innovation news |
| Society (社会) | 10,000 | Social and livelihood news |
| Current Affairs (时政) | 10,000 | Political and current events |
| Finance (财经) | 10,000 | Business and finance news |
| Real Estate (房产) | 10,000 | Real estate news |
| Home & Living (家居) | 10,000 | Home improvement news |

## 🚀 Quick Start

### Option 1: Use start script (recommended)

Windows:
```bat
start.bat
```

Linux/macOS:
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Docker

```bash
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Option 3: Manual start

Backend:
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install uv
uv pip install -e .
python main.py
```

Frontend:
```bash
cd frontend
pnpm install
pnpm run dev
```

## 🌐 URLs

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

## 📖 API

### Get categories
```http
GET /api/categories
```

### Classify a single news text
```http
POST /api/predict
Content-Type: application/json

{
  "text": "教育部发布新的课程标准"
}
```

### Batch classification
```http
POST /api/batch_predict
Content-Type: application/json

{
  "texts": ["新闻1", "新闻2", "新闻3"]
}
```

### Get model info
```http
GET /api/model_info
```

## 🧠 Model Details

- Algorithm: Multinomial Naive Bayes (MultinomialNB)
- Features: TF-IDF vectorization
- Tokenization: jieba Chinese segmenter
- Feature size: 10,000
- n-gram range: 1–2 (uni- and bi-grams)
- Train/test split: 80%/20%

## 🎨 UI Preview

Includes:

- Real-time classification with immediate results
- Confidence score display
- History of predictions for comparison
- Responsive design for desktop and mobile
- Color-coded category labels

## 📁 Project Structure

```
news-classification-system/
├── backend/                 # Backend code
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py    # Classifier implementation
│   ├── data.xlsx            # Training data
│   ├── main.py              # FastAPI app
│   └── pyproject.toml       # Python deps
├── frontend/                # Frontend code
│   ├── src/
│   │   ├── services/        # API services
│   │   ├── types/           # TypeScript types
│   │   ├── App.tsx          # Main component
│   │   └── main.tsx         # Entry
│   └── package.json         # Frontend deps
├── Dockerfile               # Docker image config
├── docker-compose.yml       # Compose config
├── start.sh                 # Linux start script
├── start.bat                # Windows start script
└── README.md                # Project docs (Chinese)
```

## 🔧 Development Guide

### Backend

1) Environment setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install -e .
```

2) Train the model
```bash
python -m models.classifier
```

3) Start dev server
```bash
python main.py
```

### Frontend

1) Install deps
```bash
cd frontend
pnpm install
```

2) Start dev server
```bash
pnpm run dev
```

3) Build production
```bash
pnpm run build
```

## 🐳 Docker

### Dockerfile notes
- Multi-stage builds to separate frontend build and backend runtime
- Small footprint with Alpine base image
- Health checks included
- Non-root user for security

### Environment variables
- PYTHONPATH: Python module path
- PYTHONUNBUFFERED: disable stdout buffering

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
pnpm run test

# End-to-end tests
pnpm run test:e2e
```

## 📈 Performance

- Persisted model cache
- Async request handling with FastAPI
- Frontend optimizations with React.memo and useMemo
- Bundling optimizations with Vite and code splitting

## 🔒 Security

- CORS configuration
- Request validation
- Container runs with least privilege

## 🐛 Troubleshooting

1) Model failed to load
- Ensure data.xlsx exists
- Verify Python dependencies are installed

2) Frontend can’t reach backend
- Ensure backend is up on port 8000
- Verify CORS settings

3) Docker build fails
- Ensure Docker is installed and running
- Check there’s enough disk space

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## 📞 Contact

Issues and suggestions: [GitHub Issues](https://github.com/zym9863/news-classification-system/issues)

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) – modern Python web framework
- [React](https://reactjs.org/) – UI library
- [Ant Design](https://ant.design/) – enterprise UI design system
- [scikit-learn](https://scikit-learn.org/) – ML library
- [jieba](https://github.com/fxsjy/jieba) – Chinese tokenizer
