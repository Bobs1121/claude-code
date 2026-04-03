#!/usr/bin/env pwsh
# Claude Code + Qwen3.5 启动脚本
# 用法:
#   .\start.ps1               # 交互模式
#   .\start.ps1 --print "xxx" # 非交互模式

$env:PATH = "C:\Users\HOZ2WX\.bun-local\node_modules\@oven\bun-windows-x64\bin;$env:PATH"

# OpenAI 兼容模式配置
$env:CLAUDE_CODE_USE_OPENAI = "1"
$env:OPENAI_BASE_URL = "http://10.190.179.61:11999/qwen3_5/v1"
$env:OPENAI_API_KEY = "tInrMSCt3fwUoJeN4j8yBNfv69+JE7D/9RDkTTk5MaUWMSIw13VONfMiOZsQqFFqCe1uwMtvoqTpSb49rrum8tq7Jfe0mPXiioUAIeVZ7zMtTkEcWH2wrfedtyx3QLyvynmMvQ/qVCflH1DTMgPmfA=="
$env:OPENAI_MODEL = "Qwen3.5-27B-FP16"
$env:ANTHROPIC_API_KEY = "dummy"

# 可选：禁用自动压缩（减少对模型的额外调用）
$env:DISABLE_AUTO_COMPACT = "1"

Set-Location $PSScriptRoot
bun run dev @args
