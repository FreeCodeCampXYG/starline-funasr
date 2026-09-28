@echo off
chcp 936 >nul
cd /d %~dp0

echo ============================================
echo  推送到 GitHub: FreeCodeCampXYG/FunASR
echo  分支: starline-asr-server
echo ============================================
echo.

git remote get-url fork >nul 2>&1
if errorlevel 1 (
    echo [信息] 添加 fork 远程...
    git remote add fork https://github.com/FreeCodeCampXYG/FunASR.git
)

echo [开始] git push fork starline-asr-server
git push fork starline-asr-server
echo.
if errorlevel 1 (
    echo [失败] 推送未成功。常见原因：
    echo   1. 网络无法访问 github.com（请确认能正常上网）
    echo   2. 需要登录：用户名填 GitHub 用户名，密码用 Personal Access Token
    echo   3. 首次可运行：git push --set-upstream fork starline-asr-server
) else (
    echo [成功] 已推送！浏览器打开：
    echo   https://github.com/FreeCodeCampXYG/FunASR/tree/starline-asr-server
)
echo.
pause
