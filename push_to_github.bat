@echo off
chcp 936 >nul
cd /d %~dp0

echo ============================================
echo  推送到 GitHub: FreeCodeCampXYG/starline-funasr
echo  目标：fork 的 main 分支（默认分支，访客直接看到）
echo ============================================
echo.

git remote get-url fork >nul 2>&1
if errorlevel 1 (
    echo [信息] 添加 fork 远程...
    git remote add fork https://github.com/FreeCodeCampXYG/starline-funasr.git
) else (
    echo [信息] 更新 fork 远程地址...
    git remote set-url fork https://github.com/FreeCodeCampXYG/starline-funasr.git
)

echo [步骤1] git fetch fork（拉取 fork 最新主线）
git fetch fork
echo [步骤2] 切到 main 并合并 fork/main（与上游基线同步，不丢东西）
git checkout main
git merge --no-edit fork/main
echo [步骤3] git push fork main
git push fork main
echo.
if errorlevel 1 (
    echo [失败] 推送未成功。常见原因：
    echo   1. 网络无法访问 github.com（请确认能正常上网）
    echo   2. 需要登录：用户名填 GitHub 用户名，密码用 Personal Access Token
    echo   3. 若提示无上游：git push --set-upstream fork main
    echo   4. 若提示非快进且不想合并：git push fork main --force
) else (
    echo [成功] 已推送到 fork 的 main 分支！浏览器打开：
    echo   https://github.com/FreeCodeCampXYG/starline-funasr
)
echo.
pause
