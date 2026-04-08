import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import aiohttp
import discord
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
LOG_FILE = BASE_DIR / "activity.log"

DEFAULT_CONFIG = {
    "token": "your_user_token_here",
    "ollama_model": "dolphin-mistral",
    "ollama_url": "http://localhost:11434/api/chat",
    "system_prompt": (
        "You are Jarvis, a helpful, witty, and intelligent Discord assistant. "
        "You remember context and reply naturally. Keep responses under 1800 characters."
    ),
    "blacklisted_servers": []
}

conversation_history = {}
MAX_HISTORY = 12

def clean_response(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def log_activity(entry: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {entry}\n"
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line)
    print(line, end="")

bot_running = False
bot_client = None
message_queue = asyncio.Queue()

async def process_queue(cfg: dict):
    while bot_running:
        try:
            message = await message_queue.get()
            await handle_message(message, cfg)
            message_queue.task_done()
        except Exception as e:
            log_activity(f"Queue error: {e}")

async def handle_message(message: discord.Message, cfg: dict):
    global client
    content_lower = message.content.lower()
    if client.user not in message.mentions and "jarvis" not in content_lower:
        return

    guild_id = str(message.guild.id) if message.guild else None
    current_cfg = load_config()
    is_me = message.author.id == client.user.id

    if guild_id and guild_id in current_cfg.get("blacklisted_servers", []) and not is_me:
        log_activity(f"BLACKLISTED: Ignored {message.author} in server {guild_id}")
        return

    prompt = message.content
    for mention in message.mentions:
        prompt = prompt.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    prompt = prompt.replace("jarvis", "").replace("Jarvis", "").replace("JARVIS", "").strip()

    if not prompt:
        prompt = "Introduce yourself briefly."

    guild_name = message.guild.name if message.guild else "DM"
    channel_id = str(message.channel.id)
    log_activity(f"[{guild_name}] {message.author}: {prompt[:100]}")
    history = conversation_history.get(channel_id, [])
    history.append({"role": "user", "content": f"{message.author}: {prompt}"})

    async with message.channel.typing():
        await asyncio.sleep(0.6)
        response = await get_ai_response(history, cfg)

    response = clean_response(response)
    log_activity(f"[{guild_name}] Jarvis: {response[:100]}")
    history.append({"role": "assistant", "content": response})
    conversation_history[channel_id] = history[-MAX_HISTORY:]

    try:
        if len(response) <= 2000:
            await message.reply(response, mention_author=False)
        else:
            chunks = [response[i:i+1990] for i in range(0, len(response), 1990)]
            for chunk in chunks:
                await message.channel.send(chunk)
                await asyncio.sleep(0.4)
    except Exception as e:
        log_activity(f"Send error: {e}")

async def get_ai_response(messages: list, cfg: dict) -> str:
    payload = {
        "model": cfg["ollama_model"],
        "messages": [{"role": "system", "content": cfg["system_prompt"]}, *messages],
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg["ollama_url"],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    return f"[Ollama HTTP Error {resp.status}]"
                data = await resp.json()
                return data.get("message", {}).get("content", "[Empty response]")
    except aiohttp.ClientConnectorError:
        return "[Ollama not running]"
    except Exception as e:
        log_activity(f"Ollama error: {e}")
        return f"[Error: {type(e).__name__}]"

async def run_bot(cfg: dict):
    global bot_running, bot_client, client
    client = discord.Client(chunk_guilds_at_startup=False)
    bot_client = client

    @client.event
    async def on_ready():
        global bot_running
        bot_running = True
        log_activity(f"✅ READY as {client.user}")
        asyncio.create_task(process_queue(cfg))

    @client.event
    async def on_message(message):
        if message.author.bot:
            return
        if client.user not in message.mentions and "jarvis" not in message.content.lower():
            return
        await message_queue.put(message)

    try:
        async with client:
            log_activity("Logging in...")
            await client.login(cfg["token"])
            await client.connect()
    except discord.LoginFailure:
        log_activity("Invalid token")
    except Exception as e:
        log_activity(f"Error: {e}")
    finally:
        bot_running = False
        log_activity("Stopped")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = BASE_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html not found</h1>")

class ConfigUpdate(BaseModel):
    token: str
    ollama_model: str
    ollama_url: str
    system_prompt: str
    blacklisted_servers: list[str] = []

class BlacklistRequest(BaseModel):
    server_id: str
    action: str

@app.get("/api/status")
async def get_status():
    guilds = []
    if bot_client and bot_running:
        try:
            guilds = [{"id": str(g.id), "name": g.name} for g in bot_client.guilds]
        except:
            pass
    return {"running": bot_running, "guilds": guilds}

@app.get("/api/logs")
async def get_logs():
    if not LOG_FILE.exists():
        return {"lines": []}
    with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-200:]
    return {"lines": lines}

@app.get("/api/config")
async def get_config_api():
    return load_config()

@app.post("/api/config")
async def update_config(new_cfg: ConfigUpdate):
    cfg = load_config()
    cfg.update({
        "token": new_cfg.token,
        "ollama_model": new_cfg.ollama_model,
        "ollama_url": new_cfg.ollama_url,
        "system_prompt": new_cfg.system_prompt,
        "blacklisted_servers": new_cfg.blacklisted_servers
    })
    save_config(cfg)
    return {"ok": True}

@app.post("/api/start")
async def start_bot():
    global bot_running
    if bot_running:
        return {"ok": False, "msg": "Already running"}
    cfg = load_config()
    if len(cfg.get("token", "")) < 50:
        return {"ok": False, "msg": "Invalid token"}
    asyncio.create_task(run_bot(cfg))
    return {"ok": True}

@app.post("/api/stop")
async def stop_bot():
    global bot_client
    if bot_client:
        await bot_client.close()
    return {"ok": True}

@app.post("/api/blacklist")
async def manage_blacklist(req: BlacklistRequest):
    cfg = load_config()
    blacklisted = set(cfg.get("blacklisted_servers", []))
    if req.action == "add":
        blacklisted.add(req.server_id)
    elif req.action == "remove":
        blacklisted.discard(req.server_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    cfg["blacklisted_servers"] = list(blacklisted)
    save_config(cfg)
    return {"ok": True}

@app.get("/favicon.ico")
async def favicon():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)