# client.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tempfile
import threading
import queue
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import pygame

LOG = logging.getLogger("donation_client")

# Constants
SUPPORTED_CURRENCIES = ["BYN", "EUR", "KZT", "RUB", "BRL", "TRY", "PLN", "UAH", "USD"]
API_BASE = "https://www.donationalerts.com/api/v1"
DONATIONS_ENDPOINT = f"{API_BASE}/alerts/donations"

DEFAULT_CONFIG = {
    "server_url": "",
    "user_id": "",
    "base_currency": "RUB",
    "min_amount_base": 50.0,
    "min_amounts": {},
    "poll_interval": 3.0,
    "tts": {
        "enabled": True,
        "piper_voice_path": "RU-dmitri.onnx",
        "gemini_api_key": "",
        "gemini_model": "gemini-2.5-flash",
        "enable_thinking": False,
        "donor_reader_delay_base": 1.8,
        "donor_reader_chars_per_sec": 20.0,
        "chunk_chars": 160,
        "min_chars": 4,
        "piper_length_scale": 1.0,
        "piper_append_silence_ms": 120,
        "system_prompt": """"""
    },
"moderation": {
        "enabled": True,
        "ban_words": [],
        "ban_action": "replace",
        "replacement_text": "",
        "log_violations": True,
        "normalize": {
            "remove_diacritics": True,
            "lowercase": True,
            "leet_map": True,
            "strip_punct": True,
            "collapse_spaces": True,
            "remove_zero_width": True,
        },
        "allow_morpho_suffixes": True,
        "fuzzy_threshold": 0.35,
        "fuzzy_min_len": 3,
    },
}

DEFAULT_CONFIG_FILE = "../config.yaml"
DEFAULT_TOKENS_FILE = "../tokens.json"
DEFAULT_SEEN_FILE = "../seen_state.json"

# Pygame init
_PYGAME_INIT_LOCK = threading.Lock()
_PYGAME_INITIALIZED = False

# Global session
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()

# Playback system
_PLAYBACK_MANAGER: Optional["PlaybackManager"] = None
_GLOBAL_CFG: Dict[str, Any] = {}

# Generation tracking
_GEN_IN_PROGRESS: set = set()
_GEN_IN_PROGRESS_LOCK = threading.Lock()
_PREPARED_TASKS: Dict[str, "PlaybackTask"] = {}
_PREPARED_LOCK = threading.Lock()
_AWAITING_SHOW: Dict[str, Dict[str, Any]] = {}
_AWAITING_LOCK = threading.Lock()

# Server auth
_SERVER_AUTH_TOKEN: Optional[str] = None
_SERVER_USER_ID: Optional[str] = None
_SERVER_AUTH_LOCK = threading.Lock()

# ============== Pygame Audio ==============

def _pygame_init() -> bool:
    global _PYGAME_INITIALIZED
    with _PYGAME_INIT_LOCK:
        if _PYGAME_INITIALIZED:
            return True
        try:
            pygame.mixer.init()
            _PYGAME_INITIALIZED = True
            return True
        except Exception:
            LOG.exception("pygame.mixer.init() failed")
            return False


def _pygame_play_file_blocking(path: str) -> None:
    ok = _pygame_init()
    if not ok:
        raise RuntimeError("pygame init failed")
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.08)
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
    except Exception:
        LOG.exception("pygame playback failed")
        raise


def play_wav_bytes(wav_bytes: bytes) -> None:
    if not wav_bytes:
        return
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes)
            tmpname = tf.name
        try:
            _pygame_play_file_blocking(tmpname)
        finally:
            try:
                os.unlink(tmpname)
            except Exception:
                pass
    except Exception:
        LOG.exception("Failed to play wav bytes")


# ============== Playback Manager ==============

@dataclass
class PlaybackTask:
    id: str
    meta: Dict[str, Any]
    chunk_queue: queue.Queue = field(default_factory=queue.Queue)
    finished_evt: threading.Event = field(default_factory=threading.Event)

    def put_chunk(self, wav_bytes: Optional[bytes]) -> None:
        self.chunk_queue.put(wav_bytes)

    def mark_done(self) -> None:
        self.finished_evt.set()
        self.chunk_queue.put(None)


class PlaybackManager:
    def __init__(self):
        self.tasks_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._playing_lock = threading.Lock()

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.tasks_q.put(None)
        self._thread.join(timeout=5.0)

    def enqueue(self, task: PlaybackTask) -> None:
        LOG.info("Enqueue playback task id=%s", task.id)
        self.tasks_q.put(task)

    def _run(self) -> None:
        LOG.info("PlaybackManager started")
        while not self._stop.is_set():
            try:
                task = self.tasks_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break
            with self._playing_lock:
                LOG.info("Starting playback for task id=%s", task.id)
                while True:
                    try:
                        chunk = task.chunk_queue.get(timeout=1.0)
                    except queue.Empty:
                        if task.finished_evt.is_set() and task.chunk_queue.empty():
                            break
                        else:
                            continue
                    if chunk is None:
                        if task.finished_evt.is_set() and task.chunk_queue.empty():
                            break
                        else:
                            continue
                    try:
                        play_wav_bytes(chunk)
                    except Exception:
                        LOG.exception("Error playing chunk for task id=%s", task.id)
                LOG.info("Finished playback for task id=%s", task.id)
        LOG.info("PlaybackManager stopped")


# ============== HTTP Session ==============

def get_session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10, pool_connections=10)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"Connection": "keep-alive", "User-Agent": "donation_client/1.0"})
        _SESSION = s
        return _SESSION


# ============== Config & State ==============

def atomic_write_json(path: str, data: Any) -> None:
    p = Path(path)
    dir_name = str(p.parent or ".")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dir_name, delete=False) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp = tf.name
    os.replace(tmp, path)


def load_json(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        LOG.exception("Failed to load JSON from %s", path)
        return default


def ensure_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        LOG.info("Config not found: creating default %s", path)
        try:
            with p.open("w", encoding="utf-8") as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False, allow_unicode=True)
        except Exception:
            LOG.exception("Failed to write default config")
    cfg = load_config(path)
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
    if "min_amounts" in cfg and isinstance(cfg["min_amounts"], dict):
        cfg["min_amounts"] = {k.upper(): float(v) for k, v in cfg["min_amounts"].items()}
    return cfg


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data
    except Exception:
        LOG.exception("Failed to load config")
        return {}


def save_tokens(path: str, tokens: Dict[str, Any]) -> None:
    atomic_write_json(path, tokens)


def load_tokens(path: str) -> Dict[str, Any]:
    return load_json(path, {})


def load_seen_state(path: str) -> Dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {}
    ids = data.get("ids", {})
    return {"ids": {str(k): v for k, v in (ids.items() if isinstance(ids, dict) else [])}}


def save_seen_state(path: str, state: Dict[str, Any]) -> None:
    atomic_write_json(path, state)


# ============== DonationAlerts API ==============

def token_expired_soon(tokens: Dict[str, Any], margin_seconds: int = 60) -> bool:
    expires_at = tokens.get("expires_at")
    if not expires_at:
        return False
    try:
        return time.time() + margin_seconds >= float(expires_at)
    except Exception:
        return False


def refresh_access_token(tokens_path: str, tokens: Dict[str, Any],
                         client_id: Optional[str], client_secret: Optional[str]) -> Dict[str, Any]:
    refresh = tokens.get("refresh_token")
    if not refresh or not client_id or not client_secret:
        LOG.debug("No refresh_token or client credentials - skipping refresh")
        return tokens
    LOG.info("Token refreshed")
    return tokens


def api_get_donations(access_token: Optional[str], timeout: int = 15) -> Optional[requests.Response]:
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    session = get_session()
    try:
        resp = session.get(DONATIONS_ENDPOINT, headers=headers, timeout=timeout)
        return resp
    except Exception as e:
        LOG.error("Network error during GET donations: %s", e)
        return None


def fetch_donations_with_refresh(tokens_path: str, tokens: Dict[str, Any],
                                 client_id: Optional[str], client_secret: Optional[str]) -> tuple:
    access = tokens.get("access_token")
    if token_expired_soon(tokens):
        LOG.info("Access token expiring soon - refreshing")
        tokens = refresh_access_token(tokens_path, tokens, client_id, client_secret)
        access = tokens.get("access_token")
    r = api_get_donations(access)
    if r is None:
        return None, tokens, access
    if r.status_code == 401:
        LOG.info("401 received - trying refresh and retry")
        tokens = refresh_access_token(tokens_path, tokens, client_id, client_secret)
        access = tokens.get("access_token")
        r = api_get_donations(access)
    return r, tokens, access


# ============== Donation Processing ==============

def normalize_amount(item: Dict[str, Any]) -> float:
    for k in ("amount_in_user_currency", "amount", "sum", "price"):
        if k in item and item[k] is not None:
            try:
                return float(item[k])
            except Exception:
                v = item[k]
                if isinstance(v, dict) and "value" in v:
                    try:
                        return float(v["value"])
                    except Exception:
                        pass
    return 0.0


def donation_currency(item: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if "amount_in_user_currency" in item and item.get("amount_in_user_currency") is not None:
        return cfg.get("base_currency", "RUB").upper()
    cur = item.get("currency") or item.get("currency_code") or ""
    cur = str(cur).upper()
    if not cur:
        return cfg.get("base_currency", "RUB").upper()
    return cur


def passes_minimum_item(item: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    min_amounts = cfg.get("min_amounts", {}) or {}
    base_currency = cfg.get("base_currency", "RUB").upper()

    if "amount_in_user_currency" in item and item.get("amount_in_user_currency") is not None:
        try:
            amount = float(item.get("amount_in_user_currency"))
        except Exception:
            amount = normalize_amount(item)
        if base_currency in min_amounts:
            try:
                return float(amount) >= float(min_amounts[base_currency])
            except Exception:
                return False
        try:
            return float(amount) >= float(cfg.get("min_amount_base", 0.0))
        except Exception:
            return False

    currency = donation_currency(item, cfg)
    amount = normalize_amount(item)
    if currency in min_amounts:
        try:
            return float(amount) >= float(min_amounts[currency])
        except Exception:
            return False
    LOG.debug("No per-currency min set for %s - rejecting donation", currency)
    return False


def compute_donor_delay(cfg: Dict[str, Any], donor_message: str) -> float:
    tcfg = cfg.get("tts", {}) or {}
    base = float(tcfg.get("donor_reader_delay_base", 1.5))
    cps = float(tcfg.get("donor_reader_chars_per_sec", 18.0))
    return base + (len(donor_message) / max(1.0, cps))


# ============== Generation Tracking ==============

def mark_gen_started(did: str) -> bool:
    with _GEN_IN_PROGRESS_LOCK:
        if did in _GEN_IN_PROGRESS:
            return False
        _GEN_IN_PROGRESS.add(did)
        return True


def mark_gen_finished(did: str) -> None:
    with _GEN_IN_PROGRESS_LOCK:
        _GEN_IN_PROGRESS.discard(did)

# ============== Server Authorization helpers ==============
def authorize_with_server(server_url: str, cfg: Dict[str, Any], tokens_path: str) -> Optional[str]:
    """
    Отправляет конфиг на сервер (/authorize), сохраняет полученный token в tokens_path.
    Возвращает token или None.
    """
    session = get_session()
    try:
        resp = session.post(f"{server_url.rstrip('/')}/authorize", json={"config": cfg}, timeout=10)
        if resp.status_code == 200:
            j = resp.json()
            token = j.get("token")
            if token:
                tokens = load_tokens(tokens_path)
                tokens["server_token"] = token
                tokens["server_user_id"] = cfg.get("user_id")
                save_tokens(tokens_path, tokens)
                with _SERVER_AUTH_LOCK:
                    global _SERVER_AUTH_TOKEN, _SERVER_USER_ID
                    _SERVER_AUTH_TOKEN = token
                    _SERVER_USER_ID = cfg.get("user_id")
                LOG.info("Authorized with server; token saved")
                return token
        LOG.error("Server /authorize failed %s: %s", resp.status_code, resp.text)
        return None
    except Exception:
        LOG.exception("Failed to authorize with server")
        return None

def load_server_token_from_tokens(tokens_path: str) -> Optional[str]:
    tokens = load_tokens(tokens_path)
    tok = tokens.get("server_token")
    uid = tokens.get("server_user_id")
    if tok and uid:
        with _SERVER_AUTH_LOCK:
            global _SERVER_AUTH_TOKEN, _SERVER_USER_ID
            _SERVER_AUTH_TOKEN = tok
            _SERVER_USER_ID = uid
        return tok
    return None

# ============== Server Communication ==============

def request_tts_from_server(server_url: str, config: Dict[str, Any],
                            item: Dict[str, Any]) -> Optional[bytes]:
    session = get_session()
    try:
        payload = {"item": item}
        with _SERVER_AUTH_LOCK:
            token = _SERVER_AUTH_TOKEN
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = session.post(
            f"{server_url.rstrip('/')}/synthesize",
            json=payload,
            timeout=120,
            headers=headers
        )
        if resp.status_code == 200:
            return resp.content
        else:
            LOG.error("Server returned status %d: %s", resp.status_code, resp.text)
            return None
    except Exception:
        LOG.exception("Failed to request TTS from server")
        return None


def schedule_enqueue(prep_task: PlaybackTask, item: Dict[str, Any],
                     cfg: Dict[str, Any], playback_mgr: PlaybackManager) -> None:
    def _delayed():
        delay = compute_donor_delay(cfg, item.get("message") or "")
        LOG.info("Delaying enqueue for prepared task id=%s by %.2f sec", prep_task.id, delay)
        time.sleep(delay)
        playback_mgr.enqueue(prep_task)

    threading.Thread(target=_delayed, daemon=True).start()


# ============== TTS Processing ==============

def process_tts_for_item(item: Dict[str, Any], cfg: Dict[str, Any],
                         playback_mgr: PlaybackManager, prefetch: bool = False) -> None:
    tcfg = cfg.get("tts", {}) or {}
    if not tcfg.get("enabled", True):
        LOG.debug("TTS disabled in config")
        return

    did = str(item.get("id") or time.time())
    task: Optional[PlaybackTask] = None
    should_enqueue_now = not prefetch

    try:
        if should_enqueue_now:
            donor_message = item.get("message") or ""
            delay = compute_donor_delay(cfg, donor_message)
            LOG.info("Live TTS: waiting %.2f sec for donor reader before enqueue (id=%s)", delay, did)
            time.sleep(delay)
            task = PlaybackTask(id=did, meta={"username": item.get("username")})
            playback_mgr.enqueue(task)
        else:
            task = PlaybackTask(id=did, meta={"username": item.get("username")})

        server_url = cfg.get("server_url", "http://localhost:5000")
        wav_data = request_tts_from_server(server_url, cfg, item)

        if wav_data:
            if task:
                task.put_chunk(wav_data)
        else:
            LOG.warning("No audio received from server for id=%s", did)

        if task:
            task.mark_done()

        if prefetch:
            with _PREPARED_LOCK:
                _PREPARED_TASKS[did] = task
            LOG.info("Prepared TTS stored for id=%s", did)

            try:
                awaiting_item = None
                with _AWAITING_LOCK:
                    awaiting_item = _AWAITING_SHOW.pop(did, None)
                if awaiting_item:
                    with _PREPARED_LOCK:
                        pre = _PREPARED_TASKS.pop(did, None)
                    if pre:
                        LOG.info("Donation id=%s was awaiting shown - scheduling prepared TTS", did)
                        schedule_enqueue(pre, awaiting_item, cfg, playback_mgr)
            except Exception:
                LOG.exception("Failed to schedule prepared TTS for awaiting donation id=%s", did)
        else:
            LOG.info("Live TTS generation finished and enqueued for id=%s", did)

    except Exception:
        LOG.exception("Exception in process_tts_for_item (id=%s)", did)
        if task is not None:
            try:
                task.mark_done()
            except Exception:
                LOG.exception("Failed to mark task done after exception (id=%s)", did)
    finally:
        mark_gen_finished(did)


# ============== Donation Event Handlers ==============

def log_arrival(item: Dict[str, Any]) -> None:
    LOG.info("ARRIVAL id=%s user=%s amount=%s %s message=%r",
             item.get("id"), item.get("username") or (item.get("user") or {}).get("name"),
             item.get("amount") or item.get("amount_in_user_currency"),
             item.get("currency"), item.get("message"))


def process_shown_donation(item: Dict[str, Any]) -> None:
    global _GLOBAL_CFG, _PLAYBACK_MANAGER
    LOG.info("SHOWN id=%s user=%s amount=%s %s message=%r",
             item.get("id"), item.get("username") or (item.get("user") or {}).get("name"),
             item.get("amount") or item.get("amount_in_user_currency"),
             item.get("currency"), item.get("message"))

    if not _GLOBAL_CFG or not _PLAYBACK_MANAGER:
        LOG.warning("Global cfg or playback manager not set; skipping TTS")
        return

    did = item.get("id")
    if did is None:
        LOG.warning("process_shown_donation: item has no id; skipping")
        return
    did_s = str(did)

    with _PREPARED_LOCK:
        pre = _PREPARED_TASKS.pop(did_s, None)
    if pre:
        LOG.info("Found prepared TTS for id=%s - scheduling enqueue", did_s)
        schedule_enqueue(pre, item, _GLOBAL_CFG, _PLAYBACK_MANAGER)
        return

    if mark_gen_started(did_s):
        thread = threading.Thread(
            target=process_tts_for_item,
            args=(item, _GLOBAL_CFG, _PLAYBACK_MANAGER, False),
            daemon=True
        )
        thread.start()
        return
    else:
        LOG.info("Generation already in progress for id=%s; waiting for prepared result", did_s)
        wait_total = 1.0
        interval = 0.25
        waited = 0.0
        while waited < wait_total:
            with _PREPARED_LOCK:
                pre = _PREPARED_TASKS.pop(did_s, None)
            if pre:
                LOG.info("Prepared TTS appeared for id=%s after waiting %.2fs", did_s, waited)
                schedule_enqueue(pre, item, _GLOBAL_CFG, _PLAYBACK_MANAGER)
                return
            time.sleep(interval)
            waited += interval

        LOG.warning("Prepared TTS did not appear for id=%s; adding to awaiting", did_s)
        try:
            with _AWAITING_LOCK:
                _AWAITING_SHOW[did_s] = item
            LOG.info("Donation id=%s added to awaiting map", did_s)
        except Exception:
            LOG.exception("Failed to add id=%s to awaiting map", did_s)


def process_poll_batch(data: List[Dict[str, Any]], state: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    ids_state = state.setdefault("ids", {})
    for item in data:
        name_val = item.get("name")
        if name_val is not None and str(name_val) != "Donations":
            LOG.debug("Skipping item because name != 'Donations': %r", name_val)
            continue

        did = item.get("id")
        if did is None:
            continue
        did_s = str(did)
        is_shown = bool(item.get("is_shown") or item.get("shown_at") or item.get("shown_at_ts"))

        if did_s not in ids_state:
            log_arrival(item)
            ids_state[did_s] = {
                "arrived_ts": int(item.get("created_at_ts") or time.time()),
                "shown": bool(is_shown),
                "shown_ts": int(item.get("shown_at_ts") or 0) if is_shown else None,
            }

            try:
                tcfg = cfg.get("tts", {}) or {}
                if tcfg.get("enabled", True) and passes_minimum_item(item, cfg):
                    if mark_gen_started(did_s):
                        thr = threading.Thread(
                            target=process_tts_for_item,
                            args=(item, cfg, _PLAYBACK_MANAGER, True),
                            daemon=True,
                        )
                        thr.start()
                    else:
                        LOG.info("Generation already in progress for id=%s - skipping prefetch", did_s)
            except Exception:
                LOG.exception("Failed to spawn prepare thread on arrival id=%s", did_s)

            if is_shown:
                with _PREPARED_LOCK:
                    pre = _PREPARED_TASKS.pop(did_s, None)
                if pre:
                    schedule_enqueue(pre, item, cfg, _PLAYBACK_MANAGER)
                else:
                    if passes_minimum_item(item, cfg):
                        process_shown_donation(item)
                    else:
                        LOG.info("Donation id=%s shown but did NOT pass minimum check", did_s)
        else:
            prev = ids_state[did_s]
            prev_shown = bool(prev.get("shown"))
            if not prev_shown and is_shown:
                ids_state[did_s]["shown"] = True
                ids_state[did_s]["shown_ts"] = int(item.get("shown_at_ts") or time.time())
                LOG.info("Detected SHOWN transition for id=%s", did_s)
                with _PREPARED_LOCK:
                    pre = _PREPARED_TASKS.pop(did_s, None)
                if pre:
                    schedule_enqueue(pre, item, cfg, _PLAYBACK_MANAGER)
                else:
                    if passes_minimum_item(item, cfg):
                        process_shown_donation(item)
                    else:
                        LOG.info("Donation id=%s now shown but did NOT pass minimum check", did_s)


# ============== Main Loop ==============

def main() -> None:
    global _GLOBAL_CFG, _PLAYBACK_MANAGER

    p = argparse.ArgumentParser(description="DonationAlerts client")
    p.add_argument("--config", "-c", default=os.getenv("DA_CONFIG", DEFAULT_CONFIG_FILE))
    p.add_argument("--tokens", "-t", default=os.getenv("DA_TOKENS_FILE", DEFAULT_TOKENS_FILE))
    p.add_argument("--state", "-s", default=os.getenv("DA_SEEN_FILE", DEFAULT_SEEN_FILE))
    p.add_argument("--client-id", default=os.getenv("DA_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.getenv("DA_CLIENT_SECRET"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    cfg = ensure_config(args.config)
    _GLOBAL_CFG = cfg
    poll_interval = float(cfg.get("poll_interval", 3.0))
    LOG.info("Config loaded. server_url=%s base_currency=%s",
             cfg.get("server_url"), cfg.get("base_currency"))

    tokens_path = args.tokens
    state_path = args.state

    tokens = load_tokens(tokens_path)
    if not tokens:
        LOG.error("Tokens not found in %s. Run oauth helper first.", tokens_path)
        sys.exit(2)

        # Try to load server_token from tokens file; if absent, perform authorization
    loaded = load_server_token_from_tokens(tokens_path)
    if not loaded:
        LOG.info("No server token found; trying to authorize with server")
        server_url = cfg.get("server_url", "http://localhost:5000")
        tok = authorize_with_server(server_url, cfg, tokens_path)
        if not tok:
            LOG.warning("Server authorization failed; continuing but TTS requests will fail until authorized")

    state = load_seen_state(state_path)
    LOG.info("Loaded state with %d tracked ids", len(state.get("ids", {})))

    playback_mgr = PlaybackManager()
    playback_mgr.start()
    _PLAYBACK_MANAGER = playback_mgr

    r, tokens, access = fetch_donations_with_refresh(tokens_path, tokens,
                                                     args.client_id, args.client_secret)
    if r and r.status_code == 200:
        try:
            j = r.json()
        except Exception:
            j = {}
        for it in j.get("data", []):
            did = it.get("id")
            if did is None:
                continue
            did_s = str(did)
            is_shown = bool(it.get("is_shown") or it.get("shown_at") or it.get("shown_at_ts"))
            if did_s not in state.setdefault("ids", {}):
                state["ids"][did_s] = {
                    "arrived_ts": int(it.get("created_at_ts") or time.time()),
                    "shown": bool(is_shown),
                    "shown_ts": int(it.get("shown_at_ts") or 0) if is_shown else None,
                }
        save_seen_state(state_path, state)
        LOG.info("Initialization complete; tracked ids=%d", len(state.get("ids", {})))
    else:
        LOG.warning("Initialization fetch failed; continuing anyway")

    consecutive_errors = 0
    try:
        while True:
            try:
                r, tokens, access = fetch_donations_with_refresh(tokens_path, tokens,
                                                                 args.client_id, args.client_secret)
                if r is None:
                    consecutive_errors += 1
                    base_wait = poll_interval * (2 ** min(consecutive_errors, 6))
                    wait = min(300, base_wait) * (0.8 + random.random() * 0.4)
                    LOG.warning("No response from server, sleeping %s sec", wait)
                    time.sleep(wait)
                    continue
                if r.status_code != 200:
                    LOG.warning("HTTP %s: %s", r.status_code, r.text)
                    consecutive_errors += 1
                    time.sleep(poll_interval)
                    continue
                consecutive_errors = 0
                try:
                    j = r.json()
                except Exception:
                    LOG.exception("Failed to decode JSON")
                    time.sleep(poll_interval)
                    continue
                data = j.get("data", [])
                if data:
                    process_poll_batch(data, state, cfg)
                    save_seen_state(state_path, state)
            except KeyboardInterrupt:
                LOG.info("Interrupted by user")
                break
            except Exception:
                LOG.exception("Exception in poll loop")
            time.sleep(poll_interval)
    finally:
        try:
            save_seen_state(state_path, state)
        except Exception:
            LOG.exception("Failed to save seen state")
        try:
            save_tokens(tokens_path, tokens)
        except Exception:
            LOG.exception("Failed to save tokens")
        try:
            playback_mgr.stop()
        except Exception:
            LOG.exception("Failed to stop playback manager")
        LOG.info("Saved state and exiting")


if __name__ == "__main__":
    main()