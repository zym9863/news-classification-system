[English Version](README_EN.md) | [中文文档](README.md)

# Intelligent News Classification System

Based on deep learning and AI technology, this system provides automatic news classification, integrated data visualization, and intelligent analysis features.

## 🚀 System Features

### Core Functions
- **🤖 AI Intelligent Classification**: Automatic news classification based on Transformer models
- **📊 Data Visualization**: Classification statistics and trend analysis with ECharts
- **📝 Intelligent Summarization**: Text summarization powered by Pollinations AI
- **🔍 Content Analysis**: In-depth AI analysis of text content and sentiment
- **📈 Real-time Statistics**: Dynamic statistics of classification data and history

### Supported Categories
- Education 📚
- Technology 💻
- Society 🏛️
- Current Affairs 🏛️
- Finance 💰
- Real Estate 🏠
- Home 🏡

## 🛠️ Tech Stack

### Frontend
- **Vue 3** + TypeScript
- **Element Plus** - UI component library
- **ECharts** - Data visualization
- **Vue Router** - Routing management
- **Axios** - HTTP client

### Backend
- **Flask** - Web framework
- **TensorFlow** - Deep learning model
- **Pollinations AI** - AI text generation
- **Pandas** - Data processing
- **Jieba** - Chinese word segmentation

### AI Integration
- **Pollinations API** - AI service without API Key
- **Transformer Model** - News classification
- **NLP** - Text preprocessing

## 📦 Installation & Run

### Requirements
- Python 3.6+
- Node.js 16+
- npm or pnpm

### Backend Installation
```bash
cd backend
pip install flask flask-cors tensorflow pandas jieba requests aiohttp
```

### Frontend Installation
```bash
cd frontend
npm install
# or use pnpm
pnpm install
```

### Start the System

#### Method 1: Use Startup Script (Windows)
```bash
start_system.bat
```

#### Method 2: Manual Start
```bash
# Start backend
cd backend
python app.py

# Start frontend (new terminal)
cd frontend
pnpm run dev
```

### Access
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## 🎯 Functional Modules

### 1. News Classification
- Input news text, AI automatically identifies category
- Supports batch classification and history
- Real-time classification statistics

### 2. Data Statistics
- Pie chart for category distribution
- Bar chart for quantity statistics
- Real-time data updates
- Data export supported

### 3. History
- View all classification history
- Search and filter supported
- Re-classification function
- Batch operations supported

### 4. AI Tools
- **Text Generation**: Generate text based on prompts
- **Intelligent Summarization**: Automatically extract key information
- **Content Analysis**: In-depth analysis of text content and sentiment

## 🔧 API Endpoints

### News Classification
```http
POST /classify
Content-Type: application/json

{
  "text": "News content"
}
```

### Get Statistics
```http
GET /stats
```

### AI Text Generation
```http
POST /ai/generate
Content-Type: application/json

{
  "prompt": "Prompt text",
  "model": "openai-large"
}
```

### AI Summarization
```http
POST /ai/summarize
Content-Type: application/json

{
  "text": "Text to summarize"
}
```

### AI Content Analysis
```http
POST /ai/analyze
Content-Type: application/json

{
  "text": "Text to analyze"
}
```

## 📊 Data Visualization

### Charts
- **Pie Chart**: Category distribution
- **Bar Chart**: Category count comparison
- **Trend Chart**: Category trend changes
- **Progress Bar**: Real-time statistics progress

### Interactive Features
- Chart zoom and pan
- Click to view details
- Real-time data updates
- Responsive design

## 🤖 AI Features

### Pollinations AI Integration
- No API Key required, free to use
- Supports multiple AI models
- Private processing, data security
- High-quality text generation

### Intelligent Analysis
- Main point extraction
- Key information identification
- Sentiment analysis
- Importance rating

## 🎨 UI Design

### Modern UI
- Responsive design, multi-device support
- Dark/Light theme switch
- Smooth animations
- Intuitive user experience

### Component Architecture
- Reusable Vue components
- Unified design language
- Modular code structure
- Easy to maintain and extend

## 🔒 Data Security

### Privacy Protection
- Local data storage
- Private AI processing
- No user data collection
- Secure API calls

## 📈 Performance Optimization

### Frontend
- Component lazy loading
- Image compression
- Code splitting
- Caching strategies

### Backend
- Model preloading
- Asynchronous processing
- Data caching
- Error handling

## 🚀 Roadmap

- [ ] Support more news categories
- [ ] Add multi-language support
- [ ] Integrate more AI models
- [ ] Add user management system
- [ ] Support batch file processing
- [ ] Mobile adaptation optimization

## 📝 Changelog

### v1.0.0 (2024-06-28)
- ✅ Basic news classification completed
- ✅ Pollinations AI integrated
- ✅ Data visualization implemented
- ✅ History management added
- ✅ AI tools developed

## 🤝 Contribution

Feel free to submit Issues and Pull Requests to improve this project!

## 📄 License

MIT License

---

**Intelligent News Classification System** - Empowering news classification with AI 🚀
