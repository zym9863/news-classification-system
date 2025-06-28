@echo off
echo ========================================
echo 智能新闻分类系统启动脚本
echo ========================================
echo.

echo 正在启动后端服务器...
cd backend
start "后端服务器" cmd /k "python app.py"
cd ..

echo 等待后端服务器启动...
timeout /t 3 /nobreak > nul

echo 正在启动前端开发服务器...
cd frontend
start "前端服务器" cmd /k "pnpm run dev"
cd ..

echo.
echo ========================================
echo 系统启动完成！
echo 后端服务器: http://localhost:5000
echo 前端服务器: http://localhost:5173
echo ========================================
echo.
echo 按任意键退出...
pause > nul
