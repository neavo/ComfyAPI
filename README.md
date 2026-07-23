# ComfyUI 指令生图服务

将自然语言指令扩写后提交到现有 ComfyUI，提供 HTTP API 和可选的 Telegram 群聊机器人。

## 准备

需要 Python 3.11+、`uv` 和可访问的 ComfyUI。先运行目标工作流，再以 **API Format** 导出到 `workflows/generation.json`。工作流必须恰好包含：

- 标题为 `API Instruction` 且具有 `inputs.text` 的文本节点。
- 标题为 `API Output` 的最终图片节点。

安装依赖：

```powershell
uv sync
```

在 `config` 目录创建以下单行 UTF-8 文本文件：

| 文件 | 内容 |
| --- | --- |
| `api_token.txt` | API Bearer Token |
| `comfy_url.txt` | ComfyUI 根地址，如 `http://127.0.0.1:8188` |
| `llm_url.txt` | OpenAI 兼容的 Chat Completions 完整地址 |
| `llm_api_key.txt` | LLM API Key |
| `llm_model.txt` | LLM 模型名 |
| `tg_bot_token.txt` | Telegram Bot Token，仅机器人需要 |

同时确保 `prompt/system.md` 非空。

## 启动

先启动 HTTP API，确认监听本机 `http://127.0.0.1:48188` 后，再启动 Telegram：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 48188
uv run python -m app.telegram
```

Windows 也可分别运行 `app_api.bat` 和 `app_tgbot.bat`。

Telegram 使用 `tg_bot_token.txt` 连接 Bot API，并使用 `api_token.txt` 作为
Bearer Token 调用本机 HTTP API；它不直连 ComfyUI 或 LLM。API 地址固定为
`http://127.0.0.1:48188`。修改端口时须同步更新 `app_api.bat`、上述启动命令和
`app/telegram.py` 中的 `GENERATION_API_URL`，无需新增配置文件。

## HTTP API

所有请求均需携带 `Authorization: Bearer TOKEN`。

### 创建任务

```http
POST /new
Content-Type: application/json
Authorization: Bearer TOKEN

{"instruction":"一只戴耳机的橘猫"}
```

`instruction` 长度为 1～4096 个字符。成功返回 HTTP 202 和任务 ID：

```json
{"id":"550e8400-e29b-41d4-a716-446655440000"}
```

### 查询结果

```http
GET /result/{id}
Authorization: Bearer TOKEN
```

完成时返回 HTTP 200 和图片；处理中返回 400，可每 2～5 秒重试。其他状态包括 401（Token 错误）、404（任务不存在）、422（参数错误）、500（生成失败）和 502（上游异常）。

### PowerShell 示例

```powershell
$headers = @{ Authorization = "Bearer TOKEN" }
$job = Invoke-RestMethod -Method Post -Uri "https://api.example.com/new" -Headers $headers `
  -ContentType "application/json" -Body (@{ instruction = "一只戴耳机的橘猫" } | ConvertTo-Json)
Invoke-RestMethod -Method Get -Uri "https://api.example.com/result/$($job.id)" `
  -Headers $headers -OutFile ".\result.image"
```

## Telegram

在 BotFather 中关闭 **Group Privacy Mode**，将机器人加入群组或超级群组，然后发送：

```text
@机器人用户名 生图描述
```

提及必须位于消息开头。任务繁忙或超时时请稍后重试；机器人重启后不会恢复尚未完成的群聊任务。

## 排错

- 配置错误：检查对应配置文件是否存在且只有一行非空内容。
- 工作流错误：重新以 API Format 导出，并检查两个节点标题。
- `LLM upstream error`：检查 LLM 地址、密钥、模型和接口兼容性。
- `ComfyUI upstream error`：检查 ComfyUI 地址、工作流及服务日志。
- 机器人看不到群消息：确认已加入群组并关闭 Group Privacy Mode。
- 机器人提示生图服务异常：确认本机 48188 端口的 API 已先启动，且
  `api_token.txt` 与 API 使用的 Token 一致。

测试：`uv run pytest`
