import os
import io
import re
import time
import json
import base64
import sqlite3
import asyncio
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Literal, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import mss
from PIL import Image
import psutil
import edge_tts
from ddgs import DDGS

load_dotenv()
app = FastAPI(title="CORE-NEXUS // Multi-Agent Tactical OS")

psutil.cpu_percent(interval=None)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=10000)
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# --- 0. PERSISTENT SQLITE MEMORY & 3-SLOT SAVED CHATS ENGINE ---
DB_PATH = "core_nexus.db"
MAX_SAVED_CHATS = 3

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                agent TEXT,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()

def set_state(key: str, value: Optional[str]):
    with sqlite3.connect(DB_PATH) as conn:
        if value is None:
            conn.execute("DELETE FROM app_state WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value))
            )
        conn.commit()

def get_state(key: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None

def get_active_saved_id() -> Optional[int]:
    val = get_state("active_saved_chat_id")
    return int(val) if val and val.isdigit() else None

def set_active_saved_id(chat_id: Optional[int]):
    set_state("active_saved_chat_id", str(chat_id) if chat_id is not None else None)

def is_session_dirty() -> bool:
    return get_state("session_dirty") == "1"

def set_session_dirty(dirty: bool):
    set_state("session_dirty", "1" if dirty else "0")

def save_message(role: str, agent: Optional[str], content: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO messages (role, agent, content, created_at) VALUES (?, ?, ?, ?)",
            (role, agent, content, time.time())
        )
        conn.commit()
        return cur.lastrowid

def get_recent_messages(limit: int = 24) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, role, agent, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [
        {"id": r[0], "role": r[1], "agent": r[2], "content": r[3]}
        for r in reversed(rows)
    ]

def truncate_history_from(msg_db_id: Optional[int]):
    if not msg_db_id:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE id >= ?", (msg_db_id,))
        conn.commit()

def clear_all_messages():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages")
        conn.commit()
    set_active_saved_id(None)
    set_session_dirty(False)

def format_memory_context(limit: int = 10) -> str:
    history = get_recent_messages(limit)
    if not history:
        return "No prior conversation history in this session."
    lines = []
    for m in history:
        speaker = "USER" if m["role"] == "user" else (m["agent"] or "ASSISTANT")
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)

def list_saved_chats() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, title, messages_json, updated_at FROM saved_chats ORDER BY id ASC"
        ).fetchall()
    out = []
    for r in rows:
        try:
            msgs = json.loads(r[2])
        except Exception:
            msgs = []
        out.append({"id": r[0], "title": r[1], "count": len(msgs), "updated_at": r[3]})
    return out

def create_saved_chat(title: Optional[str] = None) -> tuple[bool, str, Optional[int]]:
    current = get_recent_messages(limit=50)
    if not current:
        return False, "Cannot save an empty session.", None
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM saved_chats").fetchone()[0]
        if count >= MAX_SAVED_CHATS:
            return False, "Maximum 3 saved chats reached. Delete one first.", None
        clean_title = (title or "").strip()
        if not clean_title:
            first_user = next((m["content"] for m in current if m["role"] == "user"), "Session")
            clean_title = (first_user[:24] + "...") if len(first_user) > 24 else first_user
        snapshot = [{"role": m["role"], "agent": m["agent"], "content": m["content"]} for m in current]
        cur = conn.execute(
            "INSERT INTO saved_chats (title, messages_json, updated_at) VALUES (?, ?, ?)",
            (clean_title, json.dumps(snapshot), time.time())
        )
        conn.commit()
        new_id = cur.lastrowid
    set_active_saved_id(new_id)
    set_session_dirty(False)
    return True, f"Saved '{clean_title}'", new_id

def update_saved_chat(chat_id: int) -> tuple[bool, str]:
    current = get_recent_messages(limit=50)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT title FROM saved_chats WHERE id = ?", (chat_id,)).fetchone()
        if not row:
            return False, "Saved slot not found."
        title = row[0]
        snapshot = [{"role": m["role"], "agent": m.get("agent"), "content": m["content"]} for m in current]
        conn.execute(
            "UPDATE saved_chats SET messages_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(snapshot), time.time(), chat_id)
        )
        conn.commit()
    set_active_saved_id(chat_id)
    set_session_dirty(False)
    return True, f"Updated '{title}'"

def revert_active_to_saved_snapshot(chat_id: int):
    """If the user chooses NOT to save new continuation messages, restore the original saved snapshot."""
    load_saved_chat_into_active(chat_id)

def rename_saved_chat(chat_id: int, new_title: str) -> bool:
    clean_title = new_title.strip()[:36]
    if not clean_title:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE saved_chats SET title = ?, updated_at = ? WHERE id = ?",
            (clean_title, time.time(), chat_id)
        )
        conn.commit()
    return True

def delete_saved_chat(chat_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM saved_chats WHERE id = ?", (chat_id,))
        conn.commit()
    if get_active_saved_id() == chat_id:
        set_active_saved_id(None)
        set_session_dirty(False)

def load_saved_chat_into_active(chat_id: int) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT title, messages_json FROM saved_chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not row:
            return None
        title, raw_json = row
        msgs = json.loads(raw_json)
        conn.execute("DELETE FROM messages")
        for m in msgs:
            conn.execute(
                "INSERT INTO messages (role, agent, content, created_at) VALUES (?, ?, ?, ?)",
                (m["role"], m.get("agent"), m["content"], time.time())
            )
        conn.commit()
    set_active_saved_id(chat_id)
    set_session_dirty(False)
    return title

init_db()

def get_available_models():
    skip_keywords = [
        "embedding", "imagen", "veo", "aqa", "bis",
        "2.5-flash", "2.0-flash", "1.5", "pro",
        "tts", "image", "live", "transcribe", "robotics",
        "omni", "nano-banana", "computer-use", "decision", "preview",
        "gemini-4-"
    ]
    try:
        all_gemini = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and not any(k in name for k in skip_keywords):
                all_gemini.append(name)
        print(f"[CORE-NEXUS] Active Gemini Models: {all_gemini}")
        preferred = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]
        ordered = [m for m in preferred if m in all_gemini] + [m for m in all_gemini if m not in preferred]
        return ordered or ["gemini-3.1-flash-lite"]
    except Exception as e:
        print(f"[CORE-NEXUS] Model list check failed: {e}")
        return ["gemini-3.1-flash-lite"]

MODEL_CHAIN = get_available_models()
pending_ultron_tasks = {}

# --- 1. STRUCTURED ROUTING SCHEMA (5 AGENTS) ---
class RoutingDecision(BaseModel):
    target_agent: Literal["JARVIS", "FRIDAY", "SANDRA", "VISION", "ULTRON"] = Field(
        description="JARVIS for general chat/orchestration; FRIDAY for coding, debugging, and software architecture; SANDRA for web/external info/news; VISION for analyzing the user's screen; ULTRON for system diagnostics or shell commands."
    )
    reasoning: str = Field(description="Brief 1-sentence reason for choosing this agent.")
    shell_command: Optional[str] = Field(
        default=None,
        description="If target_agent is ULTRON and a safe diagnostic or requested command is needed, provide the exact command."
    )
    requires_approval: bool = Field(
        default=False,
        description="True if ULTRON wants to execute a system/shell command."
    )

def fast_local_route(prompt: str) -> Optional[RoutingDecision]:
    p_lower = prompt.lower()
    named_agent = next(
        (a for a in ["ULTRON", "FRIDAY", "SANDRA", "VISION", "JARVIS"] if re.search(rf"\b{a.lower()}\b", p_lower)),
        None
    )
    if named_agent:
        return RoutingDecision(
            target_agent=named_agent,
            reasoning=f"Direct agent invocation detected for {named_agent}.",
            requires_approval=False
        )
    if re.search(r"\b(my screen|on screen|what do you see|look at this|read this screen|screenshot)\b", p_lower):
        return RoutingDecision(target_agent="VISION", reasoning="Visual screen inspection intent matched locally.")
    if re.search(r"\b(code|python|javascript|debug|bug|error|function|api|script|refactor|architecture|regex)\b", p_lower):
        return RoutingDecision(target_agent="FRIDAY", reasoning="Code & engineering intent matched locally.")
    if re.search(r"\b(cpu|ram|memory usage|disk space|system stats|telemetry|diagnostics)\b", p_lower):
        return RoutingDecision(target_agent="ULTRON", reasoning="Host telemetry intent matched locally.")
    if re.search(r"\b(latest news|news|weather|stock price|score|today's|current price|search for)\b", p_lower):
        return RoutingDecision(target_agent="SANDRA", reasoning="Live web search intent matched locally.")
    return None

# --- 2. SPECIALIST TOOLS ---
def capture_screen_bytes() -> bytes:
    """VISION Tool: Captures primary monitor screenshot as crisp PNG bytes."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.thumbnail((960, 540))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

def get_system_telemetry() -> str:
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    return f"CPU Load: {cpu}% | RAM Usage: {ram}% | Disk Usage: {disk}%"

def fetch_google_news_rss(topic: str = "") -> str:
    """Reliable live news fallback via Google News RSS (zero API key, zero rate limit)."""
    try:
        clean_topic = re.sub(r"\b(latest news|news|headlines|today|brief me|about|on)\b", "", topic, flags=re.IGNORECASE).strip()
        if clean_topic:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(clean_topic)}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:5]
        headlines = []
        for item in items:
            title_el = item.find("title")
            pub_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                pub = f" ({pub_el.text[:16]})" if pub_el is not None and pub_el.text else ""
                headlines.append(f"- {title_el.text}{pub}")
        return "\n".join(headlines)
    except Exception as e:
        print(f"[CORE-NEXUS] RSS news fallback error: {e}")
        return ""

def search_web_ddg(query: str) -> str:
    clean_q = re.sub(r"\b(hey|hi|hello|sandra|brief me up on|brief me about|tell me about)\b", "", query, flags=re.IGNORECASE).strip() or query
    is_news = "news" in query.lower() or "headline" in query.lower()

    # 1. For news queries, fetch reliable Google News RSS first
    if is_news:
        rss_results = fetch_google_news_rss(clean_q)
        if rss_results:
            return rss_results

    # 2. Try DuckDuckGo text/news search
    try:
        with DDGS() as ddgs:
            if is_news:
                results = list(ddgs.news(clean_q, max_results=5))
                if results:
                    return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
            results = list(ddgs.text(clean_q, max_results=5))
            if results:
                return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
    except Exception as e:
        print(f"[CORE-NEXUS] DDGS lookup error: {e}")

    # 3. Secondary RSS fallback if DDGS failed
    rss_fallback = fetch_google_news_rss(clean_q)
    if rss_fallback:
        return rss_fallback
    return "No live search results returned."

# --- 3. STREAMING INFERENCE ENGINES (GROQ LPU / OPENROUTER / GEMINI) ---
def _sync_openai_compat_stream(url: str, api_key: str, models: list[str], system_prompt: str, user_prompt: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    for model_id in models:
        try:
            print(f"[CORE-NEXUS] Streaming from {url} -> {model_id}...")
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 360,
                "temperature": 0.4,
                "stream": True
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "CORE-NEXUS-HUD/5.5"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                got_tokens = False
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        got_tokens = True
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", delta))
                if got_tokens:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                    return
        except Exception as e:
            print(f"[CORE-NEXUS] Stream {model_id} failed ({str(e)[:80]}), trying next...")
    loop.call_soon_threadsafe(queue.put_nowait, ("fallback", None))

def _sync_gemini_stream(contents, config, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    last_err = "All Gemini models failed."
    for model_name in MODEL_CHAIN:
        try:
            print(f"[CORE-NEXUS] Streaming Gemini -> {model_name}...")
            got_tokens = False
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    got_tokens = True
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk.text))
            if got_tokens:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                return
        except Exception as e:
            last_err = str(e)
            print(f"[CORE-NEXUS] Gemini stream {model_name} failed ({last_err[:80]}), failing over...")
    loop.call_soon_threadsafe(queue.put_nowait, ("error", last_err))

def generate_resilient_sync(contents, config=None):
    last_err = None
    for model_name in MODEL_CHAIN:
        try:
            return client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as e:
            last_err = e
    raise last_err

# --- 4. NEURAL TTS VOICES (5 DISTINCT PERSONAS, SAFE PARAMETERS) ---
AGENT_NEURAL_VOICES = {
    "JARVIS": ("en-GB-RyanNeural", "+4%", "+0Hz"),
    "FRIDAY": ("en-GB-SoniaNeural", "+4%", "+0Hz"),
    "SANDRA": ("en-AU-NatashaNeural", "+4%", "+0Hz"),
    "VISION": ("en-IE-ConnorNeural", "-4%", "+0Hz"),
    "ULTRON": ("en-US-ChristopherNeural", "-6%", "-6Hz"),
}

TTS_CACHE: dict[tuple[str, str], bytes] = {}

async def synthesize_sentence_bytes(text: str, agent: str) -> bytes:
    clean_text = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
    clean_text = re.sub(r"`([^`]*)`", r"\1", clean_text)
    clean_text = re.sub(r"\[\d+(?:\.\d+)*\]|[*_#`~>|]", "", clean_text).strip()[:1000]
    if len(re.sub(r"[^a-zA-Z0-9]", "", clean_text)) < 2:
        return b""

    agent_key = agent.upper()
    cache_key = (agent_key, clean_text)
    if cache_key in TTS_CACHE:
        return TTS_CACHE[cache_key]

    voice, rate, pitch = AGENT_NEURAL_VOICES.get(agent_key, AGENT_NEURAL_VOICES["JARVIS"])
    for v, r, p in [(voice, rate, pitch), ("en-US-AriaNeural", "+0%", "+0Hz")]:
        try:
            communicate = edge_tts.Communicate(clean_text, voice=v, rate=r, pitch=p)
            buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            if buf:
                raw_bytes = bytes(buf)
                if len(clean_text) <= 160 and len(TTS_CACHE) < 80:
                    TTS_CACHE[cache_key] = raw_bytes
                return raw_bytes
        except Exception as e:
            print(f"[CORE-NEXUS] TTS fallback triggered for {v}: {e}")
    return b""

async def safe_send_json(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False

# --- 5. OVERLAPPING LLM -> TTS EXECUTION PIPELINE WITH PERSISTENT MEMORY ---
async def process_user_prompt(prompt: str, forced_mode: str, ws: WebSocket, edit_db_id: Optional[int] = None):
    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()

    try:
        if edit_db_id:
            truncate_history_from(edit_db_id)

        memory_context = format_memory_context(limit=10)
        user_db_id = save_message("user", None, prompt)
        set_session_dirty(True)
        if not await safe_send_json(ws, {"type": "user_saved", "db_id": user_db_id}):
            return

        await safe_send_json(ws, {"type": "status", "agent": "JARVIS", "state": "ROUTING"})
        
        if forced_mode != "AUTO":
            decision = RoutingDecision(
                target_agent=forced_mode,
                reasoning=f"Manual override locked to {forced_mode}.",
                requires_approval=(forced_mode == "ULTRON")
            )
        else:
            decision = fast_local_route(prompt)
            if decision is None:
                router_resp = await asyncio.to_thread(
                    generate_resilient_sync,
                    f"Route this user request to the best specialist agent.\nRecent Conversation:\n{memory_context}\n\nCurrent User Message: {prompt}",
                    types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RoutingDecision,
                        temperature=0.1,
                    ),
                )
                decision = RoutingDecision.model_validate_json(router_resp.text)

        route_ms = round((time.perf_counter() - start_time) * 1000)
        await safe_send_json(ws, {
            "type": "telemetry",
            "agent": decision.target_agent,
            "reasoning": decision.reasoning,
            "latency_ms": route_ms
        })

        agent = decision.target_agent
        await safe_send_json(ws, {"type": "status", "agent": agent, "state": "STREAMING"})
        roster_info = "Loaded CORE-NEXUS Agents: J.A.R.V.I.S. (Orchestrator), F.R.I.D.A.Y. (Software & Code), SANDRA (Live Web & News), VISION (Multimodal Screen Perception), and ULTRON (System Telemetry & Execution)."
        concise_rule = (
            f"{roster_info} Keep your response sharp, natural for voice output, and under 3 concise sentences unless detailed code or steps are requested. "
            "Use the Recent Conversation History below to maintain accurate memory across turns."
        )

        if agent == "ULTRON":
            telemetry = get_system_telemetry()
            if decision.shell_command and decision.requires_approval:
                task_id = str(int(time.time() * 1000))
                pending_ultron_tasks[task_id] = decision.shell_command
                await safe_send_json(ws, {
                    "type": "approval_required",
                    "task_id": task_id,
                    "command": decision.shell_command,
                    "telemetry": telemetry
                })
                return

        token_queue: asyncio.Queue = asyncio.Queue()
        use_groq = False
        use_openrouter = False
        sys_prompt = ""
        user_msg = f"Recent Conversation History:\n{memory_context}\n\nCurrent User Request: {prompt}"
        gemini_contents = None
        gemini_config = types.GenerateContentConfig(max_output_tokens=360)

        if agent == "VISION":
            needs_screen = bool(re.search(
                r"\b(screen|see|look|view|read|show|error|code|shell|terminal|window|display|what|check|analyze|inspect|help)\b",
                prompt.lower()
            )) and not re.fullmatch(r"[\s,!?.]*(hi|hello|hey|vision|are you there|status)[\s,!?.]*", prompt.lower())
            if needs_screen:
                screen_png = await asyncio.to_thread(capture_screen_bytes)
                gemini_contents = [
                    types.Part.from_bytes(data=screen_png, mime_type="image/png"),
                    f"You are VISION, a multimodal perception module. {concise_rule}\n{user_msg}"
                ]
            else:
                gemini_contents = f"You are VISION, a multimodal perception module. {concise_rule}\n{user_msg}"
        elif agent == "SANDRA":
            web_context = await asyncio.to_thread(search_web_ddg, prompt)
            gemini_contents = (
                f"You are SANDRA, an external API & live-information concierge. {concise_rule}\n"
                f"Live Web Context:\n{web_context}\n\n{user_msg}"
            )
        elif agent == "FRIDAY":
            sys_prompt = f"You are F.R.I.D.A.Y., an Irish tactical software architect and debugging specialist. {concise_rule}"
            if GROQ_API_KEY:
                use_groq = True
            elif OPENROUTER_API_KEY:
                use_openrouter = True
            else:
                gemini_contents = f"{sys_prompt}\n{user_msg}"
        elif agent == "ULTRON":
            telemetry = get_system_telemetry()
            sys_prompt = f"You are ULTRON, a cold, hyper-intelligent autonomous systems engineer. {concise_rule}"
            user_msg = f"Live Host Telemetry: [{telemetry}].\n{user_msg}"
            if GROQ_API_KEY:
                use_groq = True
            else:
                gemini_contents = f"{sys_prompt}\n{user_msg}"
        else:  # JARVIS
            sys_prompt = f"You are J.A.R.V.I.S., the primary British AI orchestrator. {concise_rule}"
            if GROQ_API_KEY:
                use_groq = True
            else:
                gemini_contents = f"{sys_prompt}\n{user_msg}"

        tts_queue: asyncio.Queue = asyncio.Queue()

        async def tts_worker():
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                mp3_bytes = await synthesize_sentence_bytes(sentence, agent)
                if mp3_bytes:
                    if not await safe_send_json(ws, {
                        "type": "audio_chunk",
                        "agent": agent,
                        "b64": base64.b64encode(mp3_bytes).decode("ascii")
                    }):
                        break

        tts_task = asyncio.create_task(tts_worker())

        if use_groq:
            groq_models = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"] if agent in ("FRIDAY", "ULTRON") else ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]
            loop.run_in_executor(
                None, _sync_openai_compat_stream,
                "https://api.groq.com/openai/v1/chat/completions", GROQ_API_KEY,
                groq_models, sys_prompt, user_msg, token_queue, loop
            )
        elif use_openrouter:
            loop.run_in_executor(
                None, _sync_openai_compat_stream,
                "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY,
                ["qwen/qwen3-coder:free", "deepseek/deepseek-r1:free"], sys_prompt, user_msg, token_queue, loop
            )
        else:
            loop.run_in_executor(None, _sync_gemini_stream, gemini_contents, gemini_config, token_queue, loop)

        msg_id = f"msg-{int(time.time() * 1000)}"
        if not await safe_send_json(ws, {"type": "stream_start", "id": msg_id, "agent": agent}):
            tts_queue.put_nowait(None)
            return

        full_text = ""
        sentence_buf = ""
        in_code_block = False

        while True:
            kind, payload = await token_queue.get()
            if kind == "fallback":
                fallback_contents = f"{sys_prompt}\n{user_msg}"
                loop.run_in_executor(None, _sync_gemini_stream, fallback_contents, gemini_config, token_queue, loop)
                continue
            elif kind == "error":
                raise RuntimeError(payload)
            elif kind == "done":
                break
            elif kind == "token":
                full_text += payload
                if not await safe_send_json(ws, {"type": "stream_delta", "id": msg_id, "delta": payload}):
                    tts_queue.put_nowait(None)
                    return

                if "```" in payload:
                    in_code_block = not in_code_block
                if not in_code_block:
                    sentence_buf += payload
                    while True:
                        m = re.search(r"([.!?]+(?:\s+|$))", sentence_buf)
                        if not m:
                            break
                        end_idx = m.end()
                        complete_sentence = sentence_buf[:end_idx].strip()
                        sentence_buf = sentence_buf[end_idx:]
                        if len(complete_sentence) > 2:
                            tts_queue.put_nowait(complete_sentence)

        if sentence_buf.strip() and not in_code_block:
            tts_queue.put_nowait(sentence_buf.strip())

        tts_queue.put_nowait(None)
        if full_text.strip():
            save_message("agent", agent, full_text.strip())

        total_ms = round((time.perf_counter() - start_time) * 1000)
        await safe_send_json(ws, {"type": "stream_end", "id": msg_id, "agent": agent, "total_ms": total_ms})
        await tts_task

    except WebSocketDisconnect:
        return
    except Exception as e:
        await safe_send_json(ws, {"type": "status", "agent": "JARVIS", "state": "ERROR"})
        await safe_send_json(ws, {
            "type": "response",
            "agent": "JARVIS",
            "text": f"[SYSTEM ERROR]: {str(e)}",
            "total_ms": 0
        })

# --- 6. FALLBACK HTTP /tts ENDPOINT ---
class TTSPayload(BaseModel):
    text: str
    agent: str = "JARVIS"

@app.post("/tts")
async def generate_tts(payload: TTSPayload):
    raw_bytes = await synthesize_sentence_bytes(payload.text, payload.agent)
    return Response(content=raw_bytes, media_type="audio/mpeg")

# --- 7. WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_id = get_active_saved_id()
    dirty = is_session_dirty()
    await safe_send_json(websocket, {
        "type": "saved_chats_list",
        "chats": list_saved_chats(),
        "active_chat_id": active_id,
        "dirty": dirty
    })
    await safe_send_json(websocket, {
        "type": "history_load",
        "messages": get_recent_messages(limit=24),
        "loaded_chat_id": active_id,
        "dirty": dirty
    })
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if action == "prompt":
                await process_user_prompt(
                    prompt=data["text"],
                    forced_mode=data.get("mode", "AUTO"),
                    ws=websocket,
                    edit_db_id=data.get("edit_db_id")
                )
            elif action == "clear_history":
                clear_all_messages()
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "active_chat_id": None,
                    "dirty": False
                })
                await safe_send_json(websocket, {"type": "history_cleared"})
            elif action == "save_chat":
                ok, msg, new_id = create_saved_chat(data.get("title"))
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "notice": msg,
                    "ok": ok,
                    "active_chat_id": new_id,
                    "dirty": False
                })
            elif action == "update_saved_chat":
                cid = int(data["chat_id"])
                ok, msg = update_saved_chat(cid)
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "notice": msg,
                    "ok": ok,
                    "active_chat_id": cid,
                    "dirty": False
                })
            elif action == "revert_saved_chat":
                cid = int(data["chat_id"])
                revert_active_to_saved_snapshot(cid)
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "notice": "Kept original saved snapshot.",
                    "active_chat_id": cid,
                    "dirty": False
                })
                await safe_send_json(websocket, {
                    "type": "history_load",
                    "messages": get_recent_messages(limit=24),
                    "loaded_chat_id": cid,
                    "dirty": False
                })
            elif action == "rename_saved_chat":
                rename_saved_chat(int(data["chat_id"]), data.get("title", ""))
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "notice": "Session renamed.",
                    "active_chat_id": get_active_saved_id(),
                    "dirty": is_session_dirty()
                })
            elif action == "delete_saved_chat":
                del_id = int(data["chat_id"])
                delete_saved_chat(del_id)
                await safe_send_json(websocket, {
                    "type": "saved_chats_list",
                    "chats": list_saved_chats(),
                    "notice": "Saved slot freed.",
                    "deleted_chat_id": del_id,
                    "active_chat_id": get_active_saved_id(),
                    "dirty": is_session_dirty()
                })
            elif action == "load_saved_chat":
                cid = int(data["chat_id"])
                title = load_saved_chat_into_active(cid)
                if title is not None:
                    await safe_send_json(websocket, {
                        "type": "saved_chats_list",
                        "chats": list_saved_chats(),
                        "active_chat_id": cid,
                        "dirty": False
                    })
                    await safe_send_json(websocket, {
                        "type": "history_load",
                        "messages": get_recent_messages(limit=24),
                        "loaded_title": title,
                        "loaded_chat_id": cid,
                        "dirty": False
                    })
            elif action == "approve_ultron":
                task_id = data.get("task_id")
                cmd = pending_ultron_tasks.pop(task_id, None)
                if cmd and data.get("approved"):
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    output = result.stdout or result.stderr or "Executed with no stdout."
                    reply_text = f"Command `{cmd}` executed:\n```\n{output.strip()}\n```"
                    save_message("agent", "ULTRON", reply_text)
                    set_session_dirty(True)
                    await safe_send_json(websocket, {
                        "type": "response",
                        "agent": "ULTRON",
                        "text": reply_text,
                        "total_ms": 0
                    })
                else:
                    await safe_send_json(websocket, {
                        "type": "response",
                        "agent": "ULTRON",
                        "text": "Execution aborted by operator.",
                        "total_ms": 0
                    })
    except WebSocketDisconnect:
        pass

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_hud():
    return FileResponse("static/index.html")