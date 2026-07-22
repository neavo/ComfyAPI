$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$token = (Get-Content "$ScriptDir\config\api_token.txt" -Raw).Trim()
$headers = @{ Authorization = "Bearer $token" }
$job = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:48188/new" `
  -Headers $headers -ContentType "application/json" `
  -Body (@{ instruction = "1girl, 角色是一个穿着女仆装、大呆毛、蓝色头发，长着鲸鱼尾巴的Q版三头身美少女, from behind, arm up, v, dark, night, cityscape, fireworks, stars in sky, shadow, snowing, pixel art" } | ConvertTo-Json)
  # -Body (@{ instruction = "A black silhouette of an elf girl sits on a swing suspended in the night sky. She has a creepy glowing smiling face. On her left a creepy speech bubble that reads 'DEEPSEEK ...?'. On her right a speech bubble 'NO! ONLY WHALE Maid !!'. She is positioned directly in front of a large, glowing crescent moon, which casts a vibrant, ethereal blue light across the rugged, dark terrain below. The ground was littered with skulls. The background is ruins, collapsed buildings. 角色是一个穿着女仆装，大呆毛，有一鲸鱼尾部的美少女, starry sky, digital art" } | ConvertTo-Json)
$job
