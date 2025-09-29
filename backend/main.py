"""
新闻分类系统后端主文件
提供新闻分类的API接口

P0 修复要点：
- 输入校验（长度、空值、批量大小）
- 统一错误响应格式与HTTP状态码
- 基础日志记录（文件+控制台）
- 安全的 CORS 配置（支持配置化）
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, constr, validator
from typing import List, Optional, Literal, Dict, Any
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
import os

from models.classifier import NewsClassifier


# =====================
# 日志配置
# =====================
def _setup_logging() -> logging.Logger:
    """配置应用日志。

    - 输出到控制台和文件（logs/app.log）
    - 轮转最大 5MB，保留 3 个备份
    """
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("app")
    if logger.handlers:
        return logger  # 避免重复添加 handler
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler("logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


logger = _setup_logging()


# =====================
# 应用与CORS
# =====================
app = FastAPI(title="新闻分类系统", description="基于机器学习的中文新闻分类API", version="1.0.0")

# CORS 允许来源（用逗号分隔），为空则退化为 "*"（并禁用 credentials）
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    allow_credentials = True
else:
    allow_origins = ["*"]
    # 若为 * 则根据规范不允许 allow_credentials=True
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# Pydantic 模型
# =====================
MAX_TEXT_LEN = 1000
MAX_BATCH_SIZE = 100


class ErrorDetail(BaseModel):
    """统一错误响应结构。"""

    code: Literal[
        "VALIDATION_ERROR",
        "PREDICTION_ERROR",
        "MODEL_NOT_READY",
        "INTERNAL_ERROR",
    ]
    message: str
    errors: Optional[List[str]] = None


class PredictRequest(BaseModel):
    """单条预测请求体。"""

    text: constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LEN) = Field(..., description="待分类文本")


class SinglePredictResponse(BaseModel):
    """单条预测响应。"""

    text: str
    predicted_category: str
    confidence: float


class BatchPredictRequest(BaseModel):
    """批量预测请求体。"""

    texts: List[constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LEN)] = Field(
        ..., description="待分类文本列表"
    )

    @validator("texts")
    def _check_batch_size(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("文本列表不能为空")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"单次最多支持 {MAX_BATCH_SIZE} 条文本")
        return v


class BatchPredictResponse(BaseModel):
    """批量预测响应。"""

    results: List[Dict[str, Any]]


class CategoriesResponse(BaseModel):
    """类别列表响应。"""

    categories: List[str]


# =====================
# 全局分类器实例与生命周期
# =====================
classifier = NewsClassifier()


@app.on_event("startup")
async def startup_event() -> None:
    """应用启动时初始化模型。

    优先尝试加载本地模型；若失败则自动训练。
    """
    try:
        if classifier.load_model():
            logger.info("模型加载成功")
        else:
            logger.warning("模型加载失败，开始训练新模型...")
            classifier.train_model()
            logger.info("模型训练完成")
    except Exception as e:
        logger.exception("启动阶段模型初始化失败")


# =====================
# 统一异常处理
# =====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求体验证失败（422）统一返回。"""
    logger.warning(f"验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(
            code="VALIDATION_ERROR",
            message="请求参数验证失败",
            errors=[e.get("msg", "参数错误") for e in exc.errors()],
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常统一返回。"""
    logger.warning(f"HTTP 错误: {exc.status_code} {exc.detail}")
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """未捕获异常统一返回（500）。"""
    logger.exception("未处理的服务器错误")
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(code="INTERNAL_ERROR", message="服务器内部错误").dict(),
    )


# =====================
# 路由
# =====================
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """健康检查端点。"""
    return {"message": "新闻分类系统API", "version": "1.0.0", "status": "运行中"}


@app.get("/api/categories", response_model=CategoriesResponse)
async def get_categories() -> CategoriesResponse:
    """获取所有新闻类别。"""
    return CategoriesResponse(categories=classifier.get_categories())


@app.post("/api/predict", response_model=SinglePredictResponse)
async def predict_single(req: PredictRequest) -> SinglePredictResponse:
    """单条新闻分类预测。"""
    try:
        result = classifier.predict(req.text)
        return SinglePredictResponse(
            text=req.text,
            predicted_category=result["category"],
            confidence=result["confidence"],
        )
    except Exception as e:
        logger.exception("单条预测失败")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(code="PREDICTION_ERROR", message=str(e)).dict(),
        )


@app.post("/api/batch_predict", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """批量新闻分类预测。"""
    try:
        results = classifier.predict_batch(req.texts)
        return BatchPredictResponse(results=results)
    except Exception as e:
        logger.exception("批量预测失败")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(code="PREDICTION_ERROR", message=str(e)).dict(),
        )


@app.get("/api/model_info")
async def get_model_info() -> Dict[str, Any]:
    """获取模型信息。"""
    return classifier.get_model_info()


# 挂载静态文件（前端构建产物）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
