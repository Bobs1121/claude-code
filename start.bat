@echo off
REM Claude Code + Qwen3.5 启动脚本
REM 用法:
REM   start.bat               -- 交互模式
REM   start.bat --print "xxx" -- 非交互模式

set PATH=C:\Users\HOZ2WX\.bun-local\node_modules\@oven\bun-windows-x64\bin;%PATH%

set CLAUDE_CODE_USE_OPENAI=1
set OPENAI_BASE_URL=http://10.190.179.61:11999/qwen3_5/v1
set OPENAI_API_KEY=tInrMSCt3fwUoJeN4j8yBNfv69+JE7D/9RDkTTk5MaUWMSIw13VONfMiOZsQqFFqCe1uwMtvoqTpSb49rrum8tq7Jfe0mPXiioUAIeVZ7zMtTkEcWH2wrfedtyx3QLyvynmMvQ/qVCflH1DTMgPmfA==
set OPENAI_MODEL=Qwen3.5-27B-FP16
set ANTHROPIC_API_KEY=dummy
set DISABLE_AUTO_COMPACT=1

cd /d "%~dp0"
bun run dev %*
