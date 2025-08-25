"""
新闻分类系统后端主文件
提供新闻分类的API接口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from models.classifier import NewsClassifier

app = FastAPI(title="新闻分类系统", description="基于机器学习的中文新闻分类API", version="1.0.0")

# 配置CORS中间件，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化分类器
classifier = NewsClassifier()

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    try:
        classifier.load_model()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("🔄 开始训练新模型...")
        classifier.train_model()
        print("✅ 模型训练完成")

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {"message": "新闻分类系统API", "version": "1.0.0", "status": "运行中"}

@app.get("/api/categories")
async def get_categories():
    """获取所有新闻类别"""
    return {"categories": classifier.get_categories()}

@app.post("/api/predict")
async def predict_single(request: dict):
    """单条新闻分类预测"""
    text = request.get("text", "")
    if not text:
        return {"error": "请提供新闻文本"}
    
    result = classifier.predict(text)
    return {
        "text": text,
        "predicted_category": result["category"],
        "confidence": result["confidence"]
    }

@app.post("/api/batch_predict") 
async def predict_batch(request: dict):
    """批量新闻分类预测"""
    texts = request.get("texts", [])
    if not texts:
        return {"error": "请提供新闻文本列表"}
    
    results = classifier.predict_batch(texts)
    return {"results": results}

@app.get("/api/model_info")
async def get_model_info():
    """获取模型信息"""
    return classifier.get_model_info()

# 挂载静态文件（前端构建产物）
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
