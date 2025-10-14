# server.py
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import threading
import queue
import io
import wave
import re
import unicodedata
import regex as _regex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response
import yaml
import tempfile
import uuid

from piper import PiperVoice
from google import genai
from google.genai import types

LOG = logging.getLogger("tts_server")

SUPPORTED_CURRENCIES = ["BYN", "EUR", "KZT", "RUB", "BRL", "TRY", "PLN", "UAH", "USD"]
DEFAULT_CONFIG = {
    # (оставляем дефолт на случай fallback, но теперь основной конфиг приходит от клиента)
    "base_currency": "RUB",
    "min_amount_base": 50.0,
    "min_amounts": {},
    "tts": {
        "enabled": True,
        "piper_voice_path": "RU-dmitri.onnx",
        "gemini_api_key": "",
        "gemini_model": "gemini-2.5-flash",
        "enable_thinking": False,
        "thinking_budget": 0,
        "chunk_chars": 160,
        "min_chars": 4,
        "piper_length_scale": 0.9,
        "piper_append_silence_ms": 120,
        "system_prompt": ""
    },
    "moderation": {}
}

# Normalization maps (как было)
_LEET_MAP = {
    "4": "а", "@": "а", "ª": "а",
    "6": "б", "8": "в",
    "3": "е", "€": "е",
    "1": "и", "!": "и", "|": "i",
    "0": "о",
    "5": "с", "$": "с",
    "7": "т",
    "2": "з",
    "9": "g",
}
_HOMOGLYPHS = {
    "a": "а", "b": "в", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х",
    "а": "a", "в": "b", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
}
_ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\uFEFF"]

# Global variables
_TTS_VOICE: Optional[PiperVoice] = None
_TTS_VOICE_LOCK = threading.Lock()
_GEMINI_CLIENT: Optional[genai.Client] = None
_GEMINI_LOCK = threading.Lock()
_BAN_PATTERNS_CACHE: Optional[List[_regex.Pattern]] = None
_BAN_WORDS_CACHE: Optional[Tuple[str, ...]] = None

# In-memory storage for authorized clients and their configs
_USER_CONFIGS: Dict[str, Dict[str, Any]] = {}
_AUTH_TOKENS: Dict[str, str] = {}  # token -> user_id
_AUTH_LOCK = threading.Lock()

app = Flask(__name__)

# ============== Normalization & Moderation ==============
# (функции normalize_for_check, build_ban_patterns, contains_banword, censor_text
#  оставлены без изменений — включаю их ниже компактно, как в оригинале)

def _remove_zero_width(s: str) -> str:
    for z in _ZERO_WIDTH:
        s = s.replace(z, "")
    return s

def _apply_leet_map(s: str) -> str:
    return "".join(_LEET_MAP.get(c, c) for c in s)

def _apply_homoglyph_map(s: str) -> str:
    out = []
    for ch in s:
        low = ch.lower()
        mapped = _HOMOGLYPHS.get(low)
        out.append(mapped if mapped is not None else ch)
    return "".join(out)

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous_row = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        current_row = [i]
        for j, ca in enumerate(a, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_for_check(s: str, cfg_normalize: dict) -> str:
    s = str(s)
    if cfg_normalize.get("remove_zero_width", True):
        s = _remove_zero_width(s)
    if cfg_normalize.get("lowercase", True):
        s = s.lower()
    if cfg_normalize.get("remove_diacritics", True):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if cfg_normalize.get("leet_map", True):
        s = _apply_leet_map(s)
    if cfg_normalize.get("strip_punct", True):
        try:
            s = _regex.sub(r"[^\p{L}\p{N}\s]", " ", s)
        except Exception:
            s = re.sub(r"[^\w\s]", " ", s)
    if cfg_normalize.get("collapse_spaces", True):
        try:
            s = _regex.sub(r"\s+", " ", s).strip()
        except Exception:
            s = re.sub(r"\s+", " ", s).strip()
    s = _apply_homoglyph_map(s)
    return s

def build_ban_patterns(ban_words: List[str], allow_morpho_suffixes: bool = True) -> List[_regex.Pattern]:
    pats: List[_regex.Pattern] = []
    char_classes = {
        "а": "[аa@4ª]", "б": "[б6]", "в": "[вv8b]", "г": "[гg]", "д": "[дd]",
        "е": "[еe3€]", "ё": "[ёеe]", "и": "[иi1!|]", "й": "[йи]", "к": "[кk]",
        "л": "[лl]", "м": "[мm]", "н": "[нh]", "о": "[оo0]", "п": "[пp]",
        "р": "[рp]", "с": "[сc5$]", "т": "[тt7]", "у": "[уy]", "ф": "[фf]",
        "х": "[хx]", "ц": "[цc]", "ч": "[чch]", "ш": "[шsh]", "щ": "[щshch]",
        "ь": "[ь']", "ы": "[ыy]", "ъ": "[ъ]", "ж": "[жj]", "з": "[з3z]",
    }
    for w in ban_words:
        w = str(w).lower().strip()
        if not w:
            continue
        pattern_chars = []
        for ch in w:
            pattern_chars.append(char_classes.get(ch, _regex.escape(ch)))
        core = "".join(pattern_chars)
        if allow_morpho_suffixes:
            pat = r"\b" + core + r"[^\s]{0,12}\b"
        else:
            pat = r"\b" + core + r"\b"
        try:
            pats.append(_regex.compile(pat, _regex.IGNORECASE))
        except Exception:
            try:
                pats.append(_regex.compile(_regex.escape(w), _regex.IGNORECASE))
            except Exception:
                continue
    return pats

def ensure_ban_patterns(cfg: Dict[str, Any]) -> List[_regex.Pattern]:
    global _BAN_PATTERNS_CACHE, _BAN_WORDS_CACHE
    mod = cfg.get("moderation", {}) or {}
    ban_words = tuple(mod.get("ban_words", []) or [])
    allow = bool(mod.get("allow_morpho_suffixes", True))
    if _BAN_WORDS_CACHE == ban_words and _BAN_PATTERNS_CACHE is not None:
        return _BAN_PATTERNS_CACHE
    _BAN_WORDS_CACHE = ban_words
    _BAN_PATTERNS_CACHE = build_ban_patterns(list(_BAN_WORDS_CACHE), allow_morpho_suffixes=allow)
    return _BAN_PATTERNS_CACHE

def contains_banword(text: str, cfg: Dict[str, Any]) -> bool:
    if not cfg.get("moderation", {}).get("enabled", True):
        return False
    mod = cfg.get("moderation", {}) or {}
    norm_cfg = mod.get("normalize", {})
    norm = normalize_for_check(text, norm_cfg)
    pats = ensure_ban_patterns(cfg)
    for p in pats:
        try:
            if p.search(norm):
                return True
        except Exception:
            continue
    fuzzy_threshold = float(mod.get("fuzzy_threshold", 0.35))
    fuzzy_min_len = int(mod.get("fuzzy_min_len", 3))
    tokens = [t for t in re.split(r"\s+", norm) if t]
    ban_words = mod.get("ban_words", []) or []
    for bw in ban_words:
        bw_n = normalize_for_check(bw, norm_cfg)
        if not bw_n:
            continue
        for tok in tokens:
            if len(tok) < fuzzy_min_len:
                continue
            if bw_n in tok or tok in bw_n:
                return True
            dist = levenshtein(bw_n, tok)
            ratio = dist / max(1, max(len(bw_n), len(tok)))
            if ratio <= fuzzy_threshold:
                return True
    return False

def censor_text(text: str, cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    mod = cfg.get("moderation", {}) or {}
    replacement = str(mod.get("replacement_text", "[запрещено]"))
    action = mod.get("ban_action", "replace")
    norm_cfg = mod.get("normalize", {})
    if not contains_banword(text, cfg):
        return text, []
    if mod.get("log_violations"):
        LOG.warning("Moderation: banned content detected in message: %r", text)
    if action == "reject":
        return replacement, ["<rejected>"]
    out = str(text)
    pats = ensure_ban_patterns(cfg)
    for p in pats:
        try:
            out = p.sub(replacement, out)
        except Exception:
            continue
    tokens = re.split(r"(\s+)", out)
    ban_words = mod.get("ban_words", []) or []
    replaced_any = False
    for i, tok in enumerate(tokens):
        if not tok or tok.isspace():
            continue
        norm_tok = normalize_for_check(tok, norm_cfg)
        for bw in ban_words:
            bw_n = normalize_for_check(bw, norm_cfg)
            if not bw_n:
                continue
            if bw_n in norm_tok or norm_tok in bw_n:
                tokens[i] = replacement
                replaced_any = True
                break
            if len(norm_tok) >= int(mod.get("fuzzy_min_len", 3)):
                dist = levenshtein(bw_n, norm_tok)
                ratio = dist / max(1, max(len(bw_n), len(norm_tok)))
                if ratio <= float(mod.get("fuzzy_threshold", 0.35)):
                    tokens[i] = replacement
                    replaced_any = True
                    break
    if replaced_any:
        return "".join(tokens), ["<replaced>"]
    return replacement, ["<replaced_all>"]

# ============== Piper TTS ==============

def load_piper_voice(path: str) -> Optional[PiperVoice]:
    global _TTS_VOICE
    with _TTS_VOICE_LOCK:
        if _TTS_VOICE is None:
            try:
                LOG.info("Loading piper voice from %s", path)
                _TTS_VOICE = PiperVoice.load(path)
            except Exception:
                LOG.exception("Failed to load Piper voice from %s", path)
                _TTS_VOICE = None
        return _TTS_VOICE

def _append_silence_to_wav_bytes(wav_bytes: bytes, ms: int = 120) -> bytes:
    if not wav_bytes or ms <= 0:
        return wav_bytes
    try:
        inp = io.BytesIO(wav_bytes)
        with wave.open(inp, "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
            nch = r.getnchannels()
            sampwidth = r.getsampwidth()
            fr = r.getframerate()
        n_silence_frames = int(fr * (ms / 1000.0))
        silence = (b"\x00" * sampwidth * nch) * n_silence_frames
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)
            w.writeframes(frames)
            w.writeframes(silence)
        return out.getvalue()
    except Exception:
        LOG.exception("Failed to append silence to wav bytes; returning original bytes")
        return wav_bytes

def synthesize_wav_bytes(voice: Optional[PiperVoice], text: str,
                         length_scale: float = 0.92, append_silence_ms: int = 120) -> Optional[bytes]:
    """
    Получить WAV-байты из PiperVoice, поддерживая несколько возможных API:
      1) voice.synthesize_wav(text) -> bytes
      2) voice.synthesize_wav(text, wav_file) where wav_file is wave.open(..., "wb")
      3) voice.synthesize(text) -> iterator of chunks (streaming API)
    Всегда возвращаем валидный WAV или None.
    """
    if voice is None:
        return None
    try:
        # 1) try direct bytes return
        try:
            res = voice.synthesize_wav(text)
            if isinstance(res, (bytes, bytearray)):
                data = bytes(res)
                try:
                    # validate it is a wav
                    with wave.open(io.BytesIO(data), "rb") as _:
                        pass
                    return _append_silence_to_wav_bytes(data, append_silence_ms)
                except Exception:
                    # not valid WAV - continue to other methods
                    LOG.debug("Direct bytes returned but not a valid WAV; trying other methods")
        except TypeError:
            # signature might not support direct-return; ignore
            pass
        except Exception:
            LOG.exception("voice.synthesize_wav(text) failed")

        # 2) try passing a Wave_write object backed by BytesIO
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                # pass the Wave_write object (this is what Piper expects)
                try:
                    voice.synthesize_wav(text, wav_file)
                except TypeError:
                    # maybe the implementation expects a syn_config kwarg
                    try:
                        # lazy import: if SynthesisConfig exists in piper
                        from piper import SynthesisConfig  # type: ignore
                        syn_cfg = SynthesisConfig(length_scale=length_scale)
                        voice.synthesize_wav(text, wav_file, syn_config=syn_cfg)
                    except Exception:
                        # re-raise original to be caught by outer except
                        raise
            data = buf.getvalue()
            if data:
                try:
                    with wave.open(io.BytesIO(data), "rb") as _:
                        pass
                    return _append_silence_to_wav_bytes(data, append_silence_ms)
                except Exception:
                    LOG.debug("BytesIO-wave path produced invalid WAV; continuing to streaming attempt")
        except Exception:
            LOG.exception("voice.synthesize_wav(text, Wave_write) attempt failed")

        # 3) try streaming API voice.synthesize(...)
        try:
            gen = voice.synthesize(text)
            buf = io.BytesIO()
            first = True
            with wave.open(buf, "wb") as wav_file:
                for chunk in gen:
                    # chunk expected to have attributes: sample_rate, sample_width,
                    # sample_channels (or sample_channels) and audio_int16_bytes (or raw bytes)
                    # adaptively read fields with fallbacks
                    sr = getattr(chunk, "sample_rate", None)
                    sw = getattr(chunk, "sample_width", None) or getattr(chunk, "sample_width_bytes", None) or getattr(chunk, "sample_width_bytes", None)
                    chs = getattr(chunk, "sample_channels", None) or getattr(chunk, "channels", None) or getattr(chunk, "sample_channels", None)
                    raw = getattr(chunk, "audio_int16_bytes", None) or getattr(chunk, "pcm16", None) or getattr(chunk, "audio_bytes", None)

                    if raw is None:
                        # maybe chunk is the raw bytes itself
                        raw = chunk if isinstance(chunk, (bytes, bytearray)) else None

                    if first:
                        if sr is None or sw is None or chs is None:
                            LOG.warning("Streaming chunk lacks audio format info; cannot build WAV")
                            break
                        try:
                            wav_file.setnchannels(int(chs))
                            wav_file.setsampwidth(int(sw))
                            wav_file.setframerate(int(sr))
                        except Exception:
                            LOG.exception("Failed to set WAV params from first streaming chunk")
                            break
                        first = False

                    if raw:
                        try:
                            wav_file.writeframes(raw)
                        except Exception:
                            LOG.exception("Failed to write frames from streaming chunk")
                            # continue trying remaining chunks

            data = buf.getvalue()
            if data:
                try:
                    with wave.open(io.BytesIO(data), "rb") as _:
                        pass
                    return _append_silence_to_wav_bytes(data, append_silence_ms)
                except Exception:
                    LOG.exception("Combined streaming chunks did not produce valid WAV")
        except Exception:
            LOG.exception("voice.synthesize (streaming) attempt failed")

        LOG.warning("PiperVoice failed to produce valid WAV for text (len=%d)", len(text))
        return None

    except Exception:
        LOG.exception("Piper synthesis failed (unexpected)")
        return None


def combine_wav_chunks(chunks: List[bytes]) -> Optional[bytes]:
    """
    Parse each WAV chunk with wave and combine PCM frames into a single valid WAV bytes.
    Requires consistent nchannels, sampwidth, framerate across chunks.
    """
    if not chunks:
        return None
    if len(chunks) == 1:
        # ensure valid wav
        try:
            with wave.open(io.BytesIO(chunks[0]), "rb") as r:
                pass
            return chunks[0]
        except Exception:
            # fallthrough to try reconstructing
            pass

    params = None
    frames_accum: List[bytes] = []
    total_frames = 0
    for idx, b in enumerate(chunks):
        try:
            with wave.open(io.BytesIO(b), "rb") as r:
                nch = r.getnchannels()
                sampwidth = r.getsampwidth()
                fr = r.getframerate()
                nframes = r.getnframes()
                data = r.readframes(nframes)
                if params is None:
                    params = (nch, sampwidth, fr)
                else:
                    if params != (nch, sampwidth, fr):
                        LOG.error("WAV chunk %d params mismatch: %s vs %s", idx, (nch, sampwidth, fr), params)
                        return None
                frames_accum.append(data)
                total_frames += nframes
        except Exception:
            LOG.exception("Failed to parse WAV chunk %d", idx)
            return None

    # write combined wav
    try:
        out = io.BytesIO()
        nch, sampwidth, fr = params
        with wave.open(out, "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)
            for f in frames_accum:
                w.writeframes(f)
        return out.getvalue()
    except Exception:
        LOG.exception("Failed to combine WAV chunks")
        return None

# ============== Gemini Client ==============
def get_gemini_client(api_key: str) -> genai.Client:
    global _GEMINI_CLIENT
    with _GEMINI_LOCK:
        if _GEMINI_CLIENT is None:
            _GEMINI_CLIENT = genai.Client(api_key=api_key)
        return _GEMINI_CLIENT

def filter_out_think_chunks(text: str, state: Dict[str, bool]) -> str:
    if not text:
        return ""
    out = []
    i = 0
    s = text
    in_think = state.get("in_think", False)
    while i < len(s):
        if not in_think:
            idx = s.find("<think>", i)
            if idx == -1:
                out.append(s[i:])
                break
            out.append(s[i:idx])
            i = idx + len("<think>")
            in_think = True
        else:
            idx = s.find("</think>", i)
            if idx == -1:
                i = len(s)
                in_think = True
            else:
                i = idx + len("</think>")
                in_think = False
    state["in_think"] = in_think
    return "".join(out)

def _split_at_boundary(text: str, max_chars: int) -> Tuple[str, str]:
    if len(text) <= max_chars:
        return text, ""
    idx = None
    for i in range(max_chars, max(0, max_chars - 40), -1):
        if i <= 0:
            break
        if text[i - 1].isspace() or text[i - 1] in ".!,;:?":
            idx = i
            break
    if idx is None:
        idx = text.rfind(" ", 0, max_chars)
    if idx is None or idx <= 0:
        idx = max_chars
    left = text[:idx].rstrip()
    rest = text[idx:].lstrip()
    return left, rest

def stream_gemini_and_synthesize(client: genai.Client, model: str, system_prompt: str,
                                 user_prompt: str, voice: Optional[PiperVoice],
                                 tcfg: Dict[str, Any], enable_thinking: bool) -> List[bytes]:
    chunks = []
    thought_state = {"in_think": False}
    buffer_parts: List[str] = []
    buffer_len = 0
    chunk_chars = int(tcfg.get("chunk_chars", 120))
    min_chars = int(tcfg.get("min_chars", 4))
    piper_length_scale = float(tcfg.get("piper_length_scale", 0.9))
    piper_append_ms = int(tcfg.get("piper_append_silence_ms", 120))

    def flush_buffer_accumulated(force: bool = False):
        nonlocal buffer_parts, buffer_len
        if not buffer_parts:
            return
        text = "".join(buffer_parts).strip()
        buffer_parts = []
        buffer_len = 0
        while text:
            left, rest = _split_at_boundary(text, chunk_chars)
            if len(left) < min_chars and not force:
                buffer_parts = [text]
                buffer_len = len(text)
                LOG.debug("Holding short fragment (len=%d) in buffer", len(left))
                return
            if not left:
                text = rest
                continue
            LOG.debug("Synthesize chunk (len=%d): %.120s", len(left), left[:120].replace("\n", " "))
            wav = synthesize_wav_bytes(voice, left, length_scale=piper_length_scale,
                                       append_silence_ms=piper_append_ms) if voice else None
            if wav:
                chunks.append(wav)
            text = rest

    try:
        config_params = {
            "system_instruction": system_prompt,
            "temperature": 0.7,
        }

        if not enable_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        response = client.models.generate_content_stream(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_params)
        )

        for chunk in response:
            try:
                content = chunk.text if hasattr(chunk, 'text') else None
                if not content:
                    continue
                good = filter_out_think_chunks(content, thought_state)
                if not good:
                    continue
                buffer_parts.append(good)
                buffer_len += len(good)
                if buffer_len >= chunk_chars:
                    flush_buffer_accumulated(force=False)
                else:
                    if good.endswith(("\n", ". ", "! ", "? ", "... ", "\n\n")) and buffer_len >= min_chars:
                        flush_buffer_accumulated(force=False)
            except Exception:
                LOG.exception("Exception handling Gemini stream chunk")

        if buffer_parts:
            flush_buffer_accumulated(force=True)

    except Exception:
        LOG.exception("Gemini streaming failed")

    return chunks

# ============== Flask Routes ==============

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/authorize', methods=['POST'])
def authorize():
    """
    Клиент отправляет: {"config": {...}}.
    Конфиг должен содержать поле "user_id".
    Сервер сохраняет config в памяти и возвращает token.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        cfg = data.get('config')
        if not cfg or not isinstance(cfg, dict):
            return jsonify({"error": "Missing config"}), 400
        user_id = str(cfg.get('user_id') or "").strip()
        if not user_id:
            return jsonify({"error": "config.user_id required"}), 400
        token = uuid.uuid4().hex
        with _AUTH_LOCK:
            _USER_CONFIGS[user_id] = cfg
            _AUTH_TOKENS[token] = user_id
        LOG.info("Authorized user_id=%s token=%s", user_id, token)
        return jsonify({"token": token}), 200
    except Exception:
        LOG.exception("Error in /authorize")
        return jsonify({"error": "internal error"}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Ожидаем Authorization: Bearer <token>
    Тело: {"item": {...}} (без поля config)
    Конфиг берём из памяти по token -> user_id -> config.
    """
    try:
        # auth token
        auth = request.headers.get("Authorization", "") or request.args.get("token", "")
        token = None
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
        elif auth:
            token = auth.strip()
        if not token:
            return jsonify({"error": "Authorization required (Bearer token)"}), 401

        with _AUTH_LOCK:
            user_id = _AUTH_TOKENS.get(token)
            cfg = _USER_CONFIGS.get(user_id) if user_id else None

        if not user_id or not cfg:
            return jsonify({"error": "Invalid token or config not found"}), 401

        data = request.get_json() or {}
        item = data.get('item', {}) or {}
        if not item:
            return jsonify({"error": "Missing item in request"}), 400

        tcfg = cfg.get('tts', {}) or {}
        if not tcfg.get('enabled', True):
            return jsonify({"error": "TTS disabled"}), 400

        api_key = tcfg.get('gemini_api_key')
        if not api_key:
            return jsonify({"error": "No Gemini API key provided in your config"}), 400

        voice_path = tcfg.get('piper_voice_path')
        voice = load_piper_voice(voice_path) if voice_path else None
        if not voice:
            return jsonify({"error": "Failed to load Piper voice"}), 500

        username = item.get('username') or (item.get('user') or {}).get('name') or "донор"
        raw_msg = item.get('message') or ""
        clean_msg, _ = censor_text(raw_msg, cfg)

        amount = item.get('amount', 0)
        currency = item.get('currency', 'RUB')

        user_prompt = (
            f"Донат (замещённый текст сообщения) от {username}: {clean_msg}\n"
            f"Сумма: {amount} {currency}\n"
            "Задача: Сформируй короткий (1-3 предложения) соркастичный, колкий ответ на донат, "
            "которая будет озвучена как реакция соведущего. "
            "Не используй запрещённые слова; если исходное сообщение содержало оскорбления — замени их. "
            "Выдавай ТОЛЬКО текст для озвучки."
        )

        system_prompt = tcfg.get('system_prompt', '')
        model = tcfg.get('gemini_model', 'gemini-2.5-flash')
        enable_thinking = tcfg.get('enable_thinking', False)

        client = get_gemini_client(api_key)

        wav_chunks = stream_gemini_and_synthesize(
            client, model, system_prompt, user_prompt, voice, tcfg, enable_thinking
        )

        if not wav_chunks:
            return jsonify({"error": "No audio generated"}), 500

        combined = combine_wav_chunks(wav_chunks)
        if not combined:
            return jsonify({"error": "Failed to combine audio chunks"}), 500

        return Response(combined, mimetype='audio/wav', headers={
            'Content-Disposition': 'attachment; filename=tts.wav'
        })

    except Exception as e:
        LOG.exception("Error in /synthesize endpoint")
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="TTS Server with Gemini API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    LOG.info("Starting TTS server on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()