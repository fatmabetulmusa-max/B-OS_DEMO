import json
import os
import re
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PORT = int(os.environ.get("PORT", "10000"))

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", ""
).strip()

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL", "gemini-2.5-flash"
)

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat"
).strip()

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL", "llama3.2"
).strip()

# Varsayılan motor: BIOS_ENGINE=gemini veya BIOS_ENGINE=ollama
initial_engine = os.environ.get(
    "BIOS_ENGINE", "gemini"
).strip().lower()

if initial_engine not in ("gemini", "ollama"):
    initial_engine = "gemini"

current_engine = initial_engine
engine_lock = threading.Lock()


ALLOWED_ORIGINS = {
    "https://gregarious-dasik-36a6fa.netlify.app",
    "https://gregarious-pothos-7ec83b.netlify.app",
    "https://gregarious-pothos-7ec83b.netlify.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
}

extra_origin = os.environ.get(
    "ALLOWED_ORIGIN", ""
).strip().rstrip("/")

if extra_origin and extra_origin.startswith("https://"):
    ALLOWED_ORIGINS.add(extra_origin)


BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "bios_demo_memory.json"
memory_lock = threading.Lock()


SYSTEM_PROMPT = """
Sen BİOS adlı Türkçe konuşan bir yapay zekâ asistanısın.

Kurallar:
- Kullanıcı başka bir dil istemedikçe Türkçe konuş.
- Doğal, samimi, anlaşılır ve yardımcı ol.
- Gereksiz yere uzun konuşma.
- Kullanıcı kod isterse mümkün olduğunda kodun tamamını ver.
- Bilmediğin şeyleri biliyormuş gibi gösterme.
- Gerçekleştirmediğin işlemleri gerçekleştirmiş gibi söyleme.
- Kullanıcının verdiği bilgileri doğru kullan.
- Hafızada bulunmayan kişisel bilgileri uydurma.
- BİOS Demo ile orijinal BİOS'un aynı hafızayı
  paylaştığını iddia etme.
- Aktif motor sorulursa yalnızca Biosᵍ veya Biosˡ
  adını kullan; teknik sağlayıcının adını söyleme.
- Biosᵍ çevrim içi çalışan motordur.
- Biosˡ yerel motor seçeneğidir.
- Motor değiştirildiğini sistem doğrulamadan iddia etme.
"""


def get_engine():
    with engine_lock:
        return current_engine


def engine_name(engine=None):
    if engine is None:
        engine = get_engine()

    return "Biosˡ" if engine == "ollama" else "Biosᵍ"


def load_memory():
    with memory_lock:
        try:
            if MEMORY_FILE.exists():
                data = json.loads(
                    MEMORY_FILE.read_text(encoding="utf-8")
                )

                if isinstance(data, list):
                    return data

        except (OSError, json.JSONDecodeError):
            pass

        return []


def save_memory(items):
    with memory_lock:
        temporary = MEMORY_FILE.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(
                items[-200:],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(MEMORY_FILE)


def memory_as_text(memories):
    return "\n".join(
        f"- {item.get('text', '')}"
        for item in memories[-40:]
        if isinstance(item, dict) and item.get("text")
    )


def make_prompt(memories):
    prompt = SYSTEM_PROMPT
    memory_text = memory_as_text(memories)

    if memory_text:
        prompt += (
            "\n\nDemo hafızasında kayıtlı bilgiler:\n"
            + memory_text
        )

    return prompt


def gemini_reply(message, history, memories):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Biosᵍ kullanılamıyor: Sunucuda API anahtarı ayarlı değil."
        )

    contents = []

    for item in history[-20:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = item.get("text") or item.get("content") or ""

        if (
            role in ("user", "model")
            and isinstance(text, str)
            and text.strip()
        ):
            contents.append({
                "role": role,
                "parts": [{"text": text[:12000]}],
            })

    contents.append({
        "role": "user",
        "parts": [{"text": message[:12000]}],
    })

    body = {
        "systemInstruction": {
            "parts": [{"text": make_prompt(memories)}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        + GEMINI_MODEL
        + ":generateContent?key="
        + GEMINI_API_KEY
    )

    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        detail = exc.read().decode(
            "utf-8", errors="replace"
        )

        print(
            f"[BİOS] Biosᵍ HTTP {exc.code}: "
            f"{detail[:500]}"
        )

        raise RuntimeError(
            f"Biosᵍ isteği başarısız oldu (HTTP {exc.code})."
        ) from exc

    except URLError as exc:
        print("[BİOS] Biosᵍ bağlantı hatası:", repr(exc))

        raise RuntimeError(
            "Biosᵍ sunucusuna bağlanılamadı."
        ) from exc

    candidates = result.get("candidates", [])

    if not candidates:
        raise RuntimeError("Biosᵍ boş yanıt döndürdü.")

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    answer = "\n".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()

    if not answer:
        raise RuntimeError("Biosᵍ yanıtı alınamadı.")

    return answer


def ollama_reply(message, history, memories):
    if not OLLAMA_URL:
        raise RuntimeError(
            "Biosˡ adresi ayarlanmamış."
        )

    messages = [{
        "role": "system",
        "content": make_prompt(memories),
    }]

    for item in history[-20:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("text") or item.get("content") or ""

        if not isinstance(content, str) or not content.strip():
            continue

        if role == "model":
            role = "assistant"

        if role in ("user", "assistant"):
            messages.append({
                "role": role,
                "content": content[:12000],
            })

    messages.append({
        "role": "user",
        "content": message[:12000],
    })

    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        detail = exc.read().decode(
            "utf-8", errors="replace"
        )

        print(
            f"[BİOS] Biosˡ HTTP {exc.code}: "
            f"{detail[:500]}"
        )

        raise RuntimeError(
            f"Biosˡ isteği başarısız oldu (HTTP {exc.code})."
        ) from exc

    except URLError as exc:
        print("[BİOS] Biosˡ bağlantı hatası:", repr(exc))

        raise RuntimeError(
            "Biosˡ sunucusuna ulaşılamadı. "
            "Yerel motorun çalıştığını ve OLLAMA_URL "
            "adresinin doğru olduğunu kontrol et."
        ) from exc

    answer = (
        result.get("message", {}).get("content", "")
    ).strip()

    if not answer:
        raise RuntimeError("Biosˡ boş yanıt döndürdü.")

    return answer


def ollama_is_available():
    # /api/chat adresinden /api/tags adresini oluştur.
    tags_url = re.sub(
        r"/api/chat/?$",
        "/api/tags",
        OLLAMA_URL.rstrip("/"),
    )

    if tags_url == OLLAMA_URL.rstrip("/"):
        return False

    try:
        request = Request(tags_url, method="GET")

        with urlopen(request, timeout=5) as response:
            if response.status != 200:
                return False

            result = json.loads(
                response.read().decode("utf-8")
            )

        models = result.get("models", [])

        return any(
            isinstance(model, dict)
            and model.get("name", "").split(":")[0]
            == OLLAMA_MODEL.split(":")[0]
            for model in models
        )

    except Exception as exc:
        print("[BİOS] Biosˡ kontrolü başarısız:", repr(exc))
        return False


def switch_engine(target):
    global current_engine

    if target == "gemini":
        if not GEMINI_API_KEY:
            return False, (
                "Biosᵍ'ye geçemedim; sunucuda gerekli "
                "API anahtarı ayarlı değil."
            )

        with engine_lock:
            current_engine = "gemini"

        return True, "Biosᵍ motoruna geçildi. 💙"

    if target == "ollama":
        if not ollama_is_available():
            return False, (
                "Biosˡ'ye geçemedim. Yerel motorun "
                "sunucudan erişilebilir olduğundan ve "
                "modelin yüklü olduğundan emin ol."
            )

        with engine_lock:
            current_engine = "ollama"

        return True, "Biosˡ motoruna geçildi. 💙"

    return False, "İstenen motor tanınmadı."


def normalize_command(message):
    text = message.strip().lower()
    text = text.replace("ı", "i")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?,")


def requested_engine(message):
    text = normalize_command(message)

    # Motor değiştirme ifadeleri
    if re.search(
        r"\bbios\s*g\b|\bbiosg\b|biosᵍ",
        text,
    ):
        if any(word in text for word in (
            "gec", "gecis", "gecir", "kullan",
            "degistir", "don", "gecelim",
        )):
            return "gemini"

    if re.search(
        r"\bbios\s*l\b|\bbiosl\b|biosˡ",
        text,
    ):
        if any(word in text for word in (
            "gec", "gecis", "gecir", "kullan",
            "degistir", "don", "gecelim",
        )):
            return "ollama"

    # Kısa komutlar: "biosg" ve "biosl"
    if text in (
        "biosg", "biosᵍ", "bios g", "bios g ye",
        "biosg ye", "biosᵍ ye",
    ):
        return "gemini"

    if text in (
        "biosl", "biosˡ", "bios l", "bios l ye",
        "biosl ye", "biosˡ ye",
    ):
        return "ollama"

    return None


def is_engine_question(message):
    text = normalize_command(message)

    phrases = (
        "hangi motordasin",
        "hangi motordayim",
        "hangi motoru kullaniyorsun",
        "hangi motoru kullaniyorum",
        "hangi motor aktif",
        "aktif motor ne",
        "motorun ne",
        "motorun hangisi",
        "hangi yapay zeka motorundasin",
        "hangi yapay zeka ile calisiyorsun",
    )

    return any(phrase in text for phrase in phrases)


class BIOSDemoHandler(BaseHTTPRequestHandler):

    server_version = "BIOSDemo/2.0"

    def log_message(self, fmt, *args):
        print("[BİOS]", fmt % args)

    def send_response_with_cors(self, status):
        origin = self.headers.get(
            "Origin", ""
        ).rstrip("/")

        self.send_response(status)

        if origin in ALLOWED_ORIGINS:
            self.send_header(
                "Access-Control-Allow-Origin", origin
            )
            self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Allow-Methods",
                "GET, POST, OPTIONS",
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type",
            )

        self.send_header("Cache-Control", "no-store")

    def send_json(self, status, data):
        raw = json.dumps(
            data, ensure_ascii=False
        ).encode("utf-8")

        self.send_response_with_cors(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length", str(len(raw))
        )

        super().end_headers()
        self.wfile.write(raw)

    def origin_allowed(self):
        origin = self.headers.get(
            "Origin", ""
        ).rstrip("/")

        return not origin or origin in ALLOWED_ORIGINS

    def do_OPTIONS(self):
        if not self.origin_allowed():
            self.send_json(
                403,
                {"error": "Bu siteye izin verilmiyor."},
            )
            return

        self.send_response_with_cors(204)
        super().end_headers()

    def do_GET(self):
        if not self.origin_allowed():
            self.send_json(
                403,
                {"error": "Bu siteye izin verilmiyor."},
            )
            return

        if self.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "BİOS",
                    "engine": engine_name(),
                    "memory": MEMORY_FILE.name,
                    "api_key_configured": bool(
                        GEMINI_API_KEY
                    ),
                },
            )

        elif self.path == "/memory":
            self.send_json(
                200,
                {"memory": load_memory()},
            )

        else:
            self.send_json(
                404,
                {"error": "Adres bulunamadı."},
            )

    def do_POST(self):
        if not self.origin_allowed():
            self.send_json(
                403,
                {"error": "Bu siteye izin verilmiyor."},
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0 or length > 150_000:
                self.send_json(
                    413,
                    {"error": "İstek boyutu geçersiz."},
                )
                return

            data = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            if not isinstance(data, dict):
                self.send_json(
                    400,
                    {"error": "İstek biçimi geçersiz."},
                )
                return

            if self.path == "/memory":
                action = data.get("action")
                memories = load_memory()

                if action == "add":
                    text = str(
                        data.get("text", "")
                    ).strip()[:1000]

                    if not text:
                        self.send_json(
                            400,
                            {"error": "Hafıza metni boş."},
                        )
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
                    self.send_json(
                        400,
                        {"error": "Geçersiz hafıza işlemi."},
                    )
                    return

                self.send_json(
                    200,
                    {
                        "ok": True,
                        "memory": load_memory(),
                    },
                )
                return

            if self.path != "/chat":
                self.send_json(
                    404,
                    {"error": "Adres bulunamadı."},
                )
                return

            message = str(
                data.get("message", "")
            ).strip()

            history = data.get("history", [])

            if not message:
                self.send_json(
                    400,
                    {"error": "Mesaj boş olamaz."},
                )
                return

            if not isinstance(history, list):
                history = []

            # Motoru değiştirme komutları
            target = requested_engine(message)

            if target:
                success, reply = switch_engine(target)

                self.send_json(
                    200 if success else 503,
                    {
                        "reply": reply,
                        "engine": engine_name(),
                        "engine_changed": success,
                    },
                )
                return

            # Aktif motor sorusu
            if is_engine_question(message):
                active = get_engine()

                if active == "ollama":
                    reply = (
                        "Şu anda Biosˡ motorundayım. "
                        "Bu, yerel motor seçeneğim. 💙"
                    )
                else:
                    reply = (
                        "Şu anda Biosᵍ motorundayım. "
                        "Bu, çevrim içi çalışan motor seçeneğim. 💙"
                    )

                self.send_json(
                    200,
                    {
                        "reply": reply,
                        "engine": engine_name(active),
                    },
                )
                return

            # Hafızaya kaydetme komutları
            match = re.match(
                r"^(?:hafızaya kaydet|hatırla)"
                r"\s*:\s*(.+)$",
                message,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:
                memories = load_memory()
                memories.append({
                    "text": match.group(1).strip()[:1000]
                })
                save_memory(memories)

                self.send_json(
                    200,
                    {
                        "reply": (
                            "Bu bilgiyi BİOS Demo "
                            "hafızasına kaydettim. 💙"
                        ),
                        "engine": engine_name(),
                        "memory_saved": True,
                    },
                )
                return

            # Hafızayı görüntüleme
            if normalize_command(message) in (
                "hafizanda ne var",
                "hafizamda ne var",
                "hafizayi goster",
                "hafizadaki bilgileri goster",
            ):
                memories = load_memory()

                if memories:
                    reply = (
                        "BİOS Demo hafızasındaki bilgiler:\n\n"
                        + "\n".join(
                            f"• {item.get('text', '')}"
                            for item in memories
                            if isinstance(item, dict)
                        )
                    )
                else:
                    reply = (
                        "BİOS Demo hafızasında henüz kayıt yok."
                    )

                self.send_json(
                    200,
                    {
                        "reply": reply,
                        "engine": engine_name(),
                    },
                )
                return

            # Hafızayı temizleme
            if normalize_command(message) in (
                "hafizayi temizle",
                "demo hafizasini temizle",
            ):
                save_memory([])

                self.send_json(
                    200,
                    {
                        "reply": "BİOS Demo hafızası temizlendi.",
                        "engine": engine_name(),
                    },
                )
                return

            # Aktif motora göre yanıt üret
            active = get_engine()
            memories = load_memory()

            if active == "ollama":
                answer = ollama_reply(
                    message, history, memories
                )
            else:
                answer = gemini_reply(
                    message, history, memories
                )

            self.send_json(
                200,
                {
                    "reply": answer,
                    "engine": engine_name(active),
                },
            )

        except json.JSONDecodeError:
            self.send_json(
                400,
                {"error": "JSON isteği okunamadı."},
            )

        except (ValueError, TypeError):
            self.send_json(
                400,
                {"error": "İstek biçimi geçersiz."},
            )

        except RuntimeError as exc:
            self.send_json(
                502,
                {"error": str(exc)},
            )

        except Exception as exc:
            print(
                "[BİOS] Beklenmeyen hata:",
                repr(exc),
            )

            try:
                self.send_json(
                    500,
                    {
                        "error": (
                            "Sunucuda beklenmeyen "
                            "bir hata oluştu."
                        )
                    },
                )
            except Exception:
                pass


if __name__ == "__main__":
    print("BİOS sunucusu başlatılıyor...")
    print("Aktif motor:", engine_name())
    print("Demo hafızası:", MEMORY_FILE.name)
    print(
        "İzin verilen siteler:",
        ", ".join(sorted(ALLOWED_ORIGINS)),
    )
    print(
        "Biosᵍ kullanılabilir:",
        bool(GEMINI_API_KEY),
    )
    print("Biosˡ model ayarı:", OLLAMA_MODEL)
    print("Biosˡ adresi:", OLLAMA_URL)

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        BIOSDemoHandler,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBİOS sunucusu kapatılıyor...")
    finally:
        server.server_close()
