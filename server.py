
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PORT = int(os.environ.get("PORT", "5000"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ALLOWED_ORIGIN = os.environ.get(
    "ALLOWED_ORIGIN",
    "https://sage-kashata-f84bc7.netlify.app"
).rstrip("/")

# Demo'nun hafızası ayrı dosyada tutulur.
BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "bios_demo_memory.json"
memory_lock = threading.Lock()

SYSTEM_PROMPT = """
Sen BİOS Demo adlı Türkçe konuşabilen bir yapay zekâ asistanısın.
Doğal, yardımcı ve anlaşılır konuş. Kullanıcı başka bir dil istemedikçe
Türkçe yanıt ver. Bilmediğin şeyleri biliyormuş gibi gösterme.
Demo'nun hafızası yalnızca bu demo sunucusuna aittir.
"""


def load_memory():
    with memory_lock:
        try:
            if MEMORY_FILE.exists():
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            pass
        return []


def save_memory(items):
    # Yalnızca bios_demo_memory.json dosyasına yazar.
    with memory_lock:
        temporary = MEMORY_FILE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(items[-200:], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        temporary.replace(MEMORY_FILE)


def gemini_reply(message, history, memories):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Sunucuda GEMINI_API_KEY ortam değişkeni ayarlanmamış."
        )

    memory_text = "\n".join(
        f"- {item.get('text', '')}"
        for item in memories[-40:]
        if isinstance(item, dict) and item.get("text")
    )
    prompt = SYSTEM_PROMPT
    if memory_text:
        prompt += "\n\nDemo hafızasındaki bilgiler:\n" + memory_text

    contents = []
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("text") or item.get("content") or ""
        if role in ("user", "model") and isinstance(text, str) and text.strip():
            contents.append({
                "role": role,
                "parts": [{"text": text[:12000]}]
            })

    contents.append({"role": "user", "parts": [{"text": message[:12000]}]})

    body = {
        "systemInstruction": {"parts": [{"text": prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + GEMINI_MODEL
        + ":generateContent?key="
        + GEMINI_API_KEY
    )
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini API hatası ({exc.code}): {detail[:700]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError("Gemini sunucusuna bağlanılamadı.") from exc

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini boş yanıt döndürdü.")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\n".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()

    if not answer:
        raise RuntimeError("Gemini yanıt metni alınamadı.")
    return answer


class BIOSDemoHandler(BaseHTTPRequestHandler):
    server_version = "BIOSDemo/1.0"

    def log_message(self, fmt, *args):
        print("[BİOS Demo]", fmt % args)

    def send_json(self, status, data):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def end_headers(self):
        # Tarayıcı preflight istekleri için CORS başlıkları.
        origin = self.headers.get("Origin", "")
        if origin == ALLOWED_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.headers.get("Origin", "") not in ("", ALLOWED_ORIGIN):
            self.send_json(403, {"error": "Bu kaynağa izin verilmiyor."})
            return

        if self.path == "/health":
            self.send_json(200, {
                "ok": True,
                "service": "BİOS Demo",
                "engine": "Gemini",
                "memory": "bios_demo_memory.json",
                "api_key_configured": bool(GEMINI_API_KEY)
            })
        elif self.path == "/memory":
            self.send_json(200, {"memory": load_memory()})
        else:
            self.send_json(404, {"error": "Adres bulunamadı."})

    def do_POST(self):
        if self.headers.get("Origin", "") not in ("", ALLOWED_ORIGIN):
            self.send_json(403, {"error": "Bu kaynağa izin verilmiyor."})
            return

        if self.path == "/memory":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    self.send_json(413, {"error": "İstek çok büyük."})
                    return
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                action = data.get("action")
                memories = load_memory()

                if action == "add":
                    text = str(data.get("text", "")).strip()[:1000]
                    if not text:
                        self.send_json(400, {"error": "Hafıza metni boş."})
                        return
                    memories.append({"text": text})
                    save_memory(memories)
                elif action == "delete":
                    index = int(data.get("index", -1))
                    if 0 <= index < len(memories):
                        memories.pop(index)
                        save_memory(memories)
                elif action == "clear":
                    save_memory([])
                else:
                    self.send_json(400, {"error": "Geçersiz hafıza işlemi."})
                    return

                self.send_json(200, {"ok": True, "memory": load_memory()})
            except (ValueError, json.JSONDecodeError):
                self.send_json(400, {"error": "Geçersiz istek."})
            return

        if self.path != "/chat":
            self.send_json(404, {"error": "Adres bulunamadı."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 150_000:
                self.send_json(413, {"error": "İstek boyutu geçersiz."})
                return

            data = json.loads(self.rfile.read(length).decode("utf-8"))
            message = str(data.get("message", "")).strip()
            history = data.get("history", [])

            if not message:
                self.send_json(400, {"error": "Mesaj boş olamaz."})
                return
            if not isinstance(history, list):
                history = []

            # Hafıza komutları yalnızca demo hafızasını etkiler.
            match = re.match(
                r"^(?:hafızaya kaydet|hatırla)\s*:\s*(.+)$",
                message,
                flags=re.IGNORECASE | re.DOTALL
            )
            if match:
                memories = load_memory()
                memories.append({"text": match.group(1).strip()[:1000]})
                save_memory(memories)
                self.send_json(200, {
                    "reply": "Bunu BİOS Demo'nun ayrı hafızasına kaydettim. 💙",
                    "engine": "Gemini",
                    "memory_saved": True
                })
                return

            if message.lower() in (
                "hafızayı temizle", "demo hafızasını temizle"
            ):
                save_memory([])
                self.send_json(200, {
                    "reply": "BİOS Demo'nun hafızası temizlendi.",
                    "engine": "Gemini"
                })
                return

            answer = gemini_reply(message, history, load_memory())
            self.send_json(200, {"reply": answer, "engine": "Gemini"})

        except json.JSONDecodeError:
            self.send_json(400, {"error": "JSON isteği okunamadı."})
        except RuntimeError as exc:
            self.send_json(502, {"error": str(exc)})
        except Exception as exc:
            print("[BİOS Demo] Beklenmeyen hata:", repr(exc))
            self.send_json(500, {"error": "Sunucuda beklenmeyen bir hata oluştu."})


if __name__ == "__main__":
    print("BİOS Demo sunucusu başlatılıyor...")
    print("Motor: Gemini")
    print("Demo hafızası:", MEMORY_FILE.name)
    print("İzin verilen site:", ALLOWED_ORIGIN)
    print("API anahtarı ayarlı:", bool(GEMINI_API_KEY))
    ThreadingHTTPServer(("0.0.0.0", PORT), BIOSDemoHandler).serve_forever()
