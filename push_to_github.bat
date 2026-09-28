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

echo [开始] git push fork main
git push fork main
echo.
if errorlevel 1 (
    echo [失败] 推送未成功。常见原因：
    echo   1. 网络无法访问 github.com（请确认能正常上网）
    echo   2. 需要登录：用户名填 GitHub 用户名，密码用 Personal Access Token
    echo   3. 首次可运行：git push --set-upstream fork main
) else (
    echo [成功] 已推送到 fork 的 main 分支！浏览器打开：
    echo   https://github.com/FreeCodeCampXYG/starline-funasr
)
echo.
pause
