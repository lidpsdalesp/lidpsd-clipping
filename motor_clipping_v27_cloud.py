# motor_clipping_v27_cloud.py - VERSÃO NUVEM (GitHub Actions)
# Requer: newspaper4k lxml[html_clean] feedparser requests beautifulsoup4 python-dotenv

import feedparser
from newspaper import Article, Config  # newspaper4k (pip install newspaper4k lxml[html_clean])

import json
import hashlib
from datetime import datetime, timedelta
import os
import nltk
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from urllib.parse import urljoin, urlparse
import re
import requests
from bs4 import BeautifulSoup
import signal
import sys
import faulthandler
faulthandler.enable()
from functools import wraps
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ==============================================================================
# 📁 ARQUIVOS DE CONFIG (TXT) - relativos ao diretório deste .py
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

ARQ_RSS = BASE_DIR / "rss_fontes.txt"
ARQ_TERMOS = BASE_DIR / "termos_monitorados.txt"
ARQ_COLUNISTAS = BASE_DIR / "colunistas.txt"
ARQ_TEMAS = BASE_DIR / "temas.txt"
ARQ_VEICULOS = BASE_DIR / "veiculos.txt"

ARQUIVO_LINKS_MANUAIS = str(BASE_DIR / "links_manuais.txt")
ARQUIVO_SAIDA = str(BASE_DIR / "fila_para_curadoria.json")


# ==============================================================================
# V26: CONTROLE DE URLs PROCESSADAS (deduplicação entre execuções)
# ==============================================================================
ARQ_URLS_PROCESSADAS = str(BASE_DIR / "urls_processadas.txt")


# Controle de URLs tentadas (para retomar após crash sem repetir a mesma URL)
ARQ_URLS_TENTADAS = str(BASE_DIR / "urls_tentadas.txt")

PULAR_GAZETA_HTML = True  # Hotfix Windows: evita crash em HTML da Gazeta do Povo

def carregar_urls_tentadas() -> set:
    p = Path(ARQ_URLS_TENTADAS)
    if not p.exists():
        return set()
    try:
        return set([line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()])
    except Exception:
        return set()

def salvar_url_tentada(url: str) -> None:
    try:
        with open(ARQ_URLS_TENTADAS, "a", encoding="utf-8") as f:
            f.write(url.strip() + chr(10))
    except Exception:
        pass


def carregar_urls_processadas() -> set:
    """Carrega o set de URLs já processadas do arquivo."""
    p = Path(ARQ_URLS_PROCESSADAS)
    if not p.exists():
        print("📝 Criando arquivo de controle de URLs...")
        return set()

    try:
        urls = set(line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"📋 {len(urls)} URLs já processadas anteriormente")
        return urls
    except Exception as e:
        print(f"⚠️  Erro ao carregar URLs processadas: {e}")
        return set()


def salvar_url_processada(url: str) -> None:
    """Registra uma URL como processada (append)."""
    try:
        with open(ARQ_URLS_PROCESSADAS, "a", encoding="utf-8") as f:
            f.write(url + "\n")
    except Exception as e:
        print(f"⚠️  Erro ao salvar URL: {e}")


def limpar_urls_antigas() -> None:
    """Auto-limpa o arquivo de controle quando crescer demais."""
    p = Path(ARQ_URLS_PROCESSADAS)
    if not p.exists():
        return

    try:
        urls = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(urls) > 10000:
            backup = p.with_suffix(p.suffix + ".backup")
            backup.write_text("\n".join(urls) + "\n", encoding="utf-8")
            urls_recentes = urls[-5000:]
            p.write_text("\n".join(urls_recentes) + "\n", encoding="utf-8")
            print(f"🧹 Arquivo de URLs limpo: {len(urls)} → {len(urls_recentes)} URLs")
            print(f"   Backup salvo em: {backup}")
    except Exception as e:
        print(f"⚠️  Erro ao limpar URLs antigas: {e}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HABILITAR_SENTIMENTO = bool(GROQ_API_KEY)


# ==============================================================================
# TIMEOUT DECORATOR (para funcoes que podem travar)
# ==============================================================================
class TimeoutError(Exception):
    pass


def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f'Timeout após {seconds}s')]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise TimeoutError(f'Função travou após {seconds}s')
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        return wrapper
    return decorator


# ==============================================================================
# --- INICIALIZAÇÃO NLTK ---
# ==============================================================================
def garantir_tokenizers_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
        return
    except LookupError:
        pass

    print("Baixando recursos de linguagem (NLTK)...")
    try:
        nltk.download("punkt", quiet=True)
        nltk.data.find("tokenizers/punkt")
        return
    except Exception:
        pass

    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


garantir_tokenizers_nltk()



# ======================================================================
# V20: SENTIMENTO
# ======================================================================
def _extrair_primeiro_json(texto: str) -> str:
    if not texto:
        raise ValueError("Resposta vazia")
    t = texto.strip()
    if "```" in t:
        partes = t.split("```")
        if len(partes) >= 3:
            t = partes[1].replace("json", "").strip()
    if t.lstrip().startswith("{") and t.rstrip().endswith("}"):
        return t.strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Sem JSON na resposta: {t[:160]}")
    return m.group(0).strip()


def analisar_sentimento_groq(texto: str, api_key: str) -> Dict[str, any]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = "Responda apenas JSON: {\"sentimento\":\"positivo|negativo|neutro\",\"confianca\":0.0,\"justificativa\":\"\"}. Texto: " + (texto or "")[:1500]
    try:
        r = requests.post(url, headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                          json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.2, "max_tokens": 220}, timeout=20)
        r.raise_for_status()
        content = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        js = _extrair_primeiro_json(content)
        out = json.loads(js)
        s = str(out.get("sentimento", "neutro")).lower().strip()
        if s not in ("positivo", "negativo", "neutro"):
            s = "neutro"
        try:
            conf = float(out.get("confianca", 0.0))
        except Exception:
            conf = 0.0
        just = str(out.get("justificativa", "") or "")
        return {"sentimento": s, "confianca": conf, "justificativa": just}
    except Exception as e:
        return {"sentimento": "neutro", "confianca": 0.0, "justificativa": f"erro: {type(e).__name__}: {str(e)[:200]}"}
def obter_emoji_sentimento(s: str) -> str:
    return {"positivo":"😊","negativo":"😟","neutro":"😐"}.get(s.lower(),"😐")
def obter_cor_sentimento(s: str) -> str:
    return {"positivo":"#10B981","negativo":"#EF4444","neutro":"#6B7280"}.get(s.lower(),"#6B7280")
# ==============================================================================
# 🧾 Leitura de TXT (ignora vazios e comentários)
# ==============================================================================
def ler_linhas_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    itens: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if s:
                itens.append(s)
    return itens


# ==============================================================================
# FONTES: RSS ou HTML (seção/listagem)
# ==============================================================================
def carregar_fontes_mistas(path: Path) -> List[Dict[str, str]]:
    linhas = ler_linhas_txt(path)
    fontes: List[Dict[str, str]] = []

    for i, linha in enumerate(linhas, start=1):
        partes = [p.strip() for p in linha.split("|")]
        if len(partes) < 2:
            raise ValueError(f"{path.name} linha {i} inválida (use 'Nome | URL' ou 'Nome | URL | HTML'): {linha}")

        nome = partes[0]
        url = partes[1]
        tipo = (partes[2].upper() if len(partes) >= 3 else "RSS")

        if not nome or not url:
            raise ValueError(f"{path.name} linha {i} inválida (nome/url vazio): {linha}")

        if tipo not in ("RSS", "HTML"):
            raise ValueError(f"{path.name} linha {i} tipo inválido (use RSS ou HTML): {linha}")

        fontes.append({"nome": nome, "url": url, "tipo": tipo})

    return fontes


def carregar_configuracoes():
    fontes = carregar_fontes_mistas(ARQ_RSS)
    termos = ler_linhas_txt(ARQ_TERMOS)
    colunistas = ler_linhas_txt(ARQ_COLUNISTAS)
    temas = ler_linhas_txt(ARQ_TEMAS)
    veiculos = ler_linhas_txt(ARQ_VEICULOS)

    if not fontes:
        raise ValueError("rss_fontes.txt está vazio.")
    if not termos:
        print("⚠️ termos_monitorados.txt está vazio: o motor não capturará nada no modo automático.")
    if not temas:
        print("⚠️ temas.txt está vazio: ainda assim o motor roda, mas 'themes' ficará só com 'Fila de Entrada'.")
    if not veiculos:
        print("⚠️ veiculos.txt está vazio: identificar_fonte ficará menos preciso.")

    return fontes, termos, colunistas, temas, veiculos


# ==============================================================================
# DATA: HOJE + ONTEM (SP)
# ==============================================================================
def now_sp() -> datetime:
    try:
        if ZoneInfo:
            return datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        pass
    return datetime.now()


def datas_permitidas() -> set:
    hoje = now_sp().date()
    ontem = hoje - timedelta(days=1)
    return {hoje.strftime("%Y-%m-%d"), ontem.strftime("%Y-%m-%d")}


def extrair_data_rss(entry) -> Optional[str]:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return time.strftime("%Y-%m-%d", entry.published_parsed)
        except Exception:
            pass

    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            return time.strftime("%Y-%m-%d", entry.updated_parsed)
        except Exception:
            pass

    return None


def extrair_data_do_url(url: str) -> Optional[str]:
    u = (url or "")

    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


# ==============================================================================
# PARÂMETROS
# ==============================================================================
MAX_NOTICIAS_POR_FONTE = 10
TIMEOUT_POR_ARTIGO = 25  # segundos
MAX_TENTATIVAS_POR_ARTIGO = 2
SALVAR_A_CADA = 1  # salva a cada notícia capturada (mais seguro em caso de travamentos)

CORES_TEMAS = [
    "#4F46E5", "#D946EF", "#F97316", "#10B981", "#EF4444", "#3B82F6",
    "#8B5CF6", "#EC4899", "#F59E0B", "#22C55E", "#DC2626", "#6366F1"
]

REQUESTS_TIMEOUT: Tuple[int, int] = (5, 20)
REQUESTS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}




# ==============================================================================
# V27.2 RESUME: filtro cedo de URLs que não são matéria
# ==============================================================================
URL_IGNORAR_SUBSTRINGS = [
    "/newsletter", "/podcasts", "/redes-sociais", "/expediente", "/mapa", "/termos-de-uso",
    "/politica-de-privacidade", "/sitemap", "/fale-conosco", "/contato", "/sobre",
    "/tag/", "/tags/", "/autor/", "/author/", "/colunistas", "/colunista",
    "/anuncie", "/assine", "/login", "/wp-login", "/wp-admin",
]

def eh_url_bloqueada(url: str) -> bool:
    u = (url or "").lower()
    return any(x in u for x in URL_IGNORAR_SUBSTRINGS)
# ==============================================================================
# Helpers de domínio / bs4
# ==============================================================================
def get_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().replace("www.", "")
    except Exception:
        return ""


def _bs_get_text(el) -> str:
    if not el:
        return ""
    return el.get_text(" ", strip=True)


def baixar_html_utf8(url: str) -> str:
    """Baixa HTML forçando UTF-8 (resolve problema da Folha)"""
    try:
        r = requests.get(url, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        
        # Força UTF-8 se o servidor não declarou encoding correto
        if r.encoding and r.encoding.lower() in ('iso-8859-1', 'latin-1', 'windows-1252'):
            r.encoding = 'utf-8'
        elif not r.encoding:
            r.encoding = 'utf-8'
        
        return r.text
    except Exception:
        return ""


# ==============================================================================
# PATCH Agência SP: legenda de fotos (Elementor)
# ==============================================================================
def extrair_legenda_agenciasp(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        el = soup.select_one(".legenda-fotos .elementor-widget-container")
        if el:
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) >= 4:
                return txt

        figcap = soup.select_one("figure figcaption")
        if figcap:
            txt = figcap.get_text(" ", strip=True)
            if txt and len(txt) >= 4:
                return txt

        for sel in (".wp-caption-text", ".caption", ".caption-text", ".credit", ".image-credit"):
            el = soup.select_one(sel)
            if el:
                txt = el.get_text(" ", strip=True)
                if txt and len(txt) >= 4:
                    return txt

        return None
    except Exception:
        return None


# ==============================================================================
# G1: autor e legenda
# ==============================================================================
def extrair_autor_g1(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for sel in (
        "meta[name='author']",
        "meta[property='article:author']",
        "meta[name='parsely-author']",
    ):
        m = soup.select_one(sel)
        if m and (m.get("content") or "").strip():
            return (m.get("content") or "").strip()

    try:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (s.string or "").strip()
            if not raw:
                continue
            data = json.loads(raw)

            nodes = []
            if isinstance(data, dict) and isinstance(data.get("@graph"), list):
                nodes = data["@graph"]
            elif isinstance(data, list):
                nodes = data
            elif isinstance(data, dict):
                nodes = [data]

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                t = str(node.get("@type", "")).lower()
                if "newsarticle" not in t and "reportagenewsarticle" not in t and "article" not in t:
                    continue

                a = node.get("author")
                if isinstance(a, dict) and (a.get("name") or "").strip():
                    return (a.get("name") or "").strip()
                if isinstance(a, list) and a and isinstance(a[0], dict) and (a[0].get("name") or "").strip():
                    return (a[0].get("name") or "").strip()
    except Exception:
        pass

    a = soup.select_one("a[href*='/autores/']")
    if a:
        return a.get_text(" ", strip=True)

    return ""


def extrair_legenda_foto_g1(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    figcap = soup.select_one("figure figcaption") or soup.select_one("figcaption")
    if figcap:
        txt = figcap.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        txt = re.sub(r"^\d+\s+de\s+\d+\s+", "", txt, flags=re.IGNORECASE).strip()
        
        if txt and len(txt) >= 4:
            return txt

    try:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (s.string or "").strip()
            if not raw:
                continue
            data = json.loads(raw)

            nodes = []
            if isinstance(data, dict) and isinstance(data.get("@graph"), list):
                nodes = data["@graph"]
            elif isinstance(data, list):
                nodes = data
            elif isinstance(data, dict):
                nodes = [data]

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                img = node.get("image")
                candidates = []
                if isinstance(img, dict):
                    candidates = [img]
                elif isinstance(img, list):
                    candidates = [x for x in img if isinstance(x, dict)]

                for im in candidates:
                    for k in ("caption", "description", "name"):
                        v = (im.get(k) or "").strip()
                        v = re.sub(r"^\d+\s+de\s+\d+\s+", "", v, flags=re.IGNORECASE).strip()
                        if v and len(v) >= 4:
                            return v
    except Exception:
        pass

    return ""


# ==============================================================================
# Folha: autor, título e legenda
# ==============================================================================
def normalizar_titulo_folha(titulo: str) -> str:
    t = (titulo or "").strip()

    if ":" in t:
        left, right = t.split(":", 1)
        if 1 <= len(left.strip()) <= 25 and right.strip():
            t = right.strip()

    t = re.sub(r"\s+", " ", t).strip()
    return t


def extrair_autor_folha(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for sel in (
        "meta[name='author']",
        "meta[property='article:author']",
        "meta[name='parsely-author']",
    ):
        m = soup.select_one(sel)
        if m and (m.get("content") or "").strip():
            return (m.get("content") or "").strip()

    a = soup.select_one("a[href*='/autores/']")
    if a:
        return a.get_text(" ", strip=True)

    return ""


def extrair_legenda_foto_folha(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    fig = soup.select_one("figcaption.widget-image__subtitle")
    if fig:
        credits_el = fig.select_one("span.widget-image__credits")
        credits_txt = credits_el.get_text(" ", strip=True) if credits_el else ""
        if credits_el:
            credits_el.extract()

        caption_txt = fig.get_text(" ", strip=True)
        caption_txt = re.sub(r"\s*-\s*$", "", caption_txt).strip()

        if caption_txt and credits_txt:
            return f"{caption_txt} (Foto: {credits_txt})"
        return caption_txt or credits_txt

    figcap = soup.select_one("figure figcaption") or soup.select_one("figcaption")
    if not figcap:
        return ""

    txt = figcap.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()

    if " - " in txt:
        partes = [p.strip() for p in txt.split(" - ") if p.strip()]
        if len(partes) >= 2:
            legenda = partes[0]
            credito = " - ".join(partes[1:])
            return f"{legenda} (Foto: {credito})"

    return txt


# ==============================================================================
# Gazeta do Povo: override title/legenda + dedupe Foto:
# ==============================================================================
def extrair_override_gazeta_do_povo(html: str) -> Dict[str, str]:
    out: Dict[str, str] = {"title": "", "imageCaption": ""}

    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("div.postLayout_post-title__hT_aC h1") or soup.select_one("h1")
    title = _bs_get_text(h1)
    if title and len(title) >= 6:
        out["title"] = title

    cap = soup.select_one("span.postMainImage_post-main-image-caption__8y_dS") or soup.select_one("span[class*='caption']")
    caption_txt = _bs_get_text(cap).strip()

    credit_txt = ""
    try:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (s.string or "").strip()
            if not raw:
                continue
            data = json.loads(raw)

            nodes = []
            if isinstance(data, dict) and isinstance(data.get("@graph"), list):
                nodes = data["@graph"]
            elif isinstance(data, list):
                nodes = data
            elif isinstance(data, dict):
                nodes = [data]

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if node.get("@type") not in ("NewsArticle", "Article"):
                    continue

                img = node.get("image")
                if isinstance(img, dict):
                    if not caption_txt:
                        caption_txt = (img.get("caption") or "").strip()
                    credit_txt = (img.get("creditText") or "").strip()
    except Exception:
        pass

    caption_txt = (caption_txt or "").strip()
    credit_txt = (credit_txt or "").strip()

    def limpar_foto_duplicada(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\)\s*\(", ") (", s).strip()
        s = re.sub(r"(\(Foto:\s*[^)]+\))\s*\1\b", r"\1", s, flags=re.IGNORECASE).strip()
        return s

    if caption_txt and re.search(r"\(Foto:\s*[^)]+\)", caption_txt, flags=re.IGNORECASE):
        out["imageCaption"] = limpar_foto_duplicada(caption_txt)
        return out

    if caption_txt and credit_txt:
        out["imageCaption"] = limpar_foto_duplicada(f"{caption_txt} (Foto: {credit_txt})")
    elif caption_txt:
        out["imageCaption"] = limpar_foto_duplicada(caption_txt)
    elif credit_txt:
        out["imageCaption"] = limpar_foto_duplicada(f"Foto: {credit_txt}")

    return out


# ==============================================================================
# Poder360: override definitivo (JSON-LD + meta tags + fallback visual)
# ==============================================================================
def extrair_override_poder360(html: str) -> Dict[str, str]:
    out: Dict[str, str] = {"title": "", "subtitle": "", "author": "", "imageCaption": ""}

    soup = BeautifulSoup(html, "html.parser")

    def meta_content(selector: str) -> str:
        m = soup.select_one(selector)
        return (m.get("content") or "").strip() if m else ""

    try:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (s.string or "").strip()
            if not raw:
                continue
            data = json.loads(raw)

            nodes = []
            if isinstance(data, dict) and isinstance(data.get("@graph"), list):
                nodes = data["@graph"]
            elif isinstance(data, list):
                nodes = data
            elif isinstance(data, dict):
                nodes = [data]

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                t = str(node.get("@type", "")).lower()
                if not any(x in t for x in ("newsarticle", "reportagenewsarticle", "article")):
                    continue

                if not out["title"]:
                    out["title"] = (node.get("headline") or "").strip()
                if not out["subtitle"]:
                    out["subtitle"] = (node.get("description") or "").strip()

                if not out["author"]:
                    a = node.get("author")
                    if isinstance(a, dict):
                        out["author"] = (a.get("name") or "").strip()
                    elif isinstance(a, list) and a and isinstance(a[0], dict):
                        out["author"] = (a[0].get("name") or "").strip()
    except Exception:
        pass

    if not out["title"]:
        out["title"] = meta_content("meta[property='og:title']") or meta_content("meta[name='twitter:title']")
    if not out["subtitle"]:
        out["subtitle"] = meta_content("meta[name='description']") or meta_content("meta[property='og:description']")
    if not out["author"]:
        out["author"] = meta_content("meta[name='author']") or meta_content("meta[name='twitter:data1']")

    if out["author"]:
        out["author"] = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out["author"]).strip()

    if not out["title"]:
        h1 = soup.select_one("h1.inner-page-section__title") or soup.select_one("h1")
        out["title"] = _bs_get_text(h1).strip()

    if not out["subtitle"]:
        h2p = soup.select_one("h2.inner-page-section__line p")
        if h2p:
            out["subtitle"] = _bs_get_text(h2p).strip()
        else:
            h2 = soup.select_one("h2.inner-page-section__line")
            out["subtitle"] = _bs_get_text(h2).strip()

    figcap = soup.select_one("figcaption.inner-page-section__caption") or soup.select_one("figure figcaption")
    cap = _bs_get_text(figcap).strip()
    if cap:
        out["imageCaption"] = cap

    return out


# ==============================================================================
# URL regex por domínio (para fontes HTML)
# ==============================================================================
def regex_materia_por_url_base(url_origem: str) -> Optional[re.Pattern]:
    u = (url_origem or "").lower()

    if "agenciasp.sp.gov.br" in u:
        return re.compile(r"^https?://(www\.)?agenciasp\.sp\.gov\.br/[^/]+/?$")

    if "al.sp.gov.br" in u:
        return re.compile(r"^https?://(www\.)?al\.sp\.gov\.br/noticia/\?id=\d+")

    if "correiodamanha.com.br" in u:
        return re.compile(r"^https?://(www\.)?correiodamanha\.com\.br/.*/\d{4}/\d{2}/\d{6}-")

    if "gazetadopovo.com.br" in u:
        return re.compile(r"^https?://(www\.)?gazetadopovo\.com\.br/[^?#]+$")

    if "poder360.com.br" in u:
        return re.compile(r"^https?://(www\.)?poder360\.com\.br/[^?#]+$")

    return None


# ==============================================================================
# RSS download com timeout real
# ==============================================================================
def baixar_rss_bytes(url: str, tentativas: int = 2) -> Optional[bytes]:
    last_err = None
    for t in range(1, tentativas + 1):
        try:
            r = requests.get(url, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            print(f"      ⚠️ Falha ao baixar RSS (tentativa {t}/{tentativas}): {e}")
            time.sleep(0.8)

    print(f"      ❌ RSS indisponível após {tentativas} tentativas: {last_err}")
    return None


# ==============================================================================
# HTML seção/listagem -> links
# ==============================================================================


def extrair_links_de_secao(url_secao: str, limite: int = 120) -> List[str]:
    r = requests.get(url_secao, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    base = f"{urlparse(url_secao).scheme}://{urlparse(url_secao).netloc}"
    base_netloc = urlparse(base).netloc

    rx_materia = regex_materia_por_url_base(url_secao)

    links: List[str] = []
    vistos = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full = urljoin(base, href)
        p = urlparse(full)

        if p.netloc and p.netloc != base_netloc:
            continue
        if p.scheme not in ("http", "https"):
            continue

        if "#" in full:
            full = full.split("#", 1)[0]

        fulll = full.lower()

        if eh_url_bloqueada(fulll):
            continue

        if fulll in base.rstrip("/"):
            continue

        if fulll.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf")):
            continue

        if rx_materia and (not rx_materia.search(full)):
            continue

        if full in vistos:
            continue

        vistos.add(full)
        links.append(full)

        if len(links) >= limite:
            break

    return links


# ==============================================================================
# Conversores/Utils
# ==============================================================================
def converter_colunistas(lista_nomes: List[str]):
    return [{"id": f"col-gen-{i}", "name": nome, "photo": ""} for i, nome in enumerate(lista_nomes)]


def converter_temas(lista_nomes: List[str]):
    return [{"name": nome, "icon": "fas fa-tag", "color": CORES_TEMAS[i % len(CORES_TEMAS)], "subtemas": []}
            for i, nome in enumerate(lista_nomes)]


def safe_str(valor) -> str:
    if valor is None:
        return ""
    return str(valor).strip()


def identificar_fonte(url: str, lista_veiculos: List[str]) -> str:
    url_lower = (url or "").lower()

    # DGABC: não confundir com 'ABC (Espanha)' por slug curto
    if ('dgabc.com.br' in url_lower) or ('dgabc.com' in url_lower):
        return 'Diário do Grande ABC'

    mapa_rapido = {
        "g1.globo": "G1",
        "folha.uol": "Folha de S.Paulo",
        "estadao.com": "Estadão",
        "al.sp.gov": "Portal Alesp",
        "cnnbrasil": "CNN Brasil",
        "agenciabrasil": "Agência Brasil",
        "gazetadopovo": "Gazeta do Povo",
        "poder360.com": "Poder360",
        "sp.gov.br": "Agência SP",
        "uol.com": "UOL",
        "r7.com": "R7",
        "bbc.co": "BBC News Brasil",
        "veja.abril": "Veja",
        "exame.com": "Exame",
        "infomoney": "InfoMoney",
        "moneytimes": "Money Times",
        "camara.leg": "Câmara dos Deputados",
        "senado.leg": "Senado",
        "dw.com": "Deutsche Welle (DW) Brasil",
        "elpais.com": "El País Brasil",
    }

    for dominio, nome_real in mapa_rapido.items():
        if dominio in url_lower:
            return nome_real

    for veiculo in lista_veiculos:
        slug = veiculo.lower().replace(" ", "").replace(".", "").replace("-", "")
        slug_clean = (slug.replace("(espanha)", "")
                        .replace("(portugal)", "")
                        .replace("(eua)", "")
                        .replace("(argentina)", "")
                        .replace("(reino unido)", "")
                        .replace("(colombia)", "")
                        .replace("(alemanha)", ""))
        # Evita falsos positivos com siglas muito curtas (ex.: 'abc' dentro de 'dgabc')
        if len(slug_clean) < 4:
            continue
        if slug_clean and slug_clean in url_lower.replace("-", "").replace(".", ""):
            return veiculo

    return "Outros"


def eh_relevante(texto: str, termos_monitorados: List[str]):
    texto_l = (texto or "").lower()
    for termo in termos_monitorados:
        if termo.lower() in texto_l:
            return True, termo
    return False, None


# ==============================================================================
# EXTRAÇÃO DE NOTÍCIA COM TIMEOUT E RETRY
# ==============================================================================
@timeout_decorator(TIMEOUT_POR_ARTIGO)

# ======================================================================
# V20: FILTRO
# ======================================================================
def eh_pagina_indice(titulo: str, url: str) -> bool:
    if not titulo or len(titulo) < 15:
        return True
    t, u = titulo.lower(), url.lower()

    # DGABC: páginas de índice/tags costumam trazer esse texto genérico
    if ('dgabc' in u) and ('notícias e informações do grande abc' in t):
        return True
    padroes_titulo = ["últimas notícias", "notícias sobre", "tudo sobre", " | folha tópicos", "editoriais | folha", "newsletter", "assine"]
    if any(p in t for p in padroes_titulo):
        return True
    padroes_url = ["/newsletter", "/podcasts", "/redes-sociais/", "/expediente/", "/mapa/", "/termos-de-uso/", "/autor/", "/author/", "/colunistas", "/colunista"]
    if any(p in u for p in padroes_url):
        return True
    secoes = ["/vozes/", "/opiniao/editoriais/", "/opiniao/", "/saber/", "/ultimas-noticias/", "/vida-e-cidadania/", "/economia/", "/republica/", "/ideias/", "/mundo/", "/cultura/", "/parana/", "/educacao/", "/bomgourmet/", "/haus/", "/poder/", "/mercado/", "/cotidiano/", "/ilustrada/", "/esporte/", "/tec/", "/equilibrio/", "/comida/", "/turismo/", "/saopaulo/"]
    for s in secoes:
        if u.endswith(s) or u.endswith(s.rstrip("/")):
            return True
    if "|" in titulo:
        if len(titulo.split("|")[0].strip()) < 25:
            return True
    return False

def _extrair_noticia_core(
    url: str,
    lista_veiculos: List[str],
    termos_monitorados: List[str],
    origem_manual: Optional[str],
    verificar_relevancia: bool,
    data_rss: Optional[str]
):
    # Download com UTF-8 forçado
    html_bruto = baixar_html_utf8(url)

    config_news = Config()
    config_news.browser_user_agent = REQUESTS_HEADERS["User-Agent"]
    config_news.request_timeout = 15

    article = Article(url, config=config_news, language="pt")
    # Usa o HTML UTF-8 que já baixamos
    if not html_bruto:
        return None

    article.download(input_html=html_bruto)
    article.html = html_bruto
    article.parse()

    # --- LIMPEZA DGABC (título com sufixos e texto/keywords de índice) ---
    dominio_tmp = get_domain(url)
    if 'dgabc' in (dominio_tmp or ''):
        # Tenta usar og:title como base (às vezes é mais estável que o title do newspaper)
        try:
            soup_dg = BeautifulSoup(html_bruto, 'html.parser')
            ogt = soup_dg.select_one("meta[property='og:title']")
            if ogt and (ogt.get('content') or '').strip():
                article.title = (ogt.get('content') or '').strip()
        except Exception:
            pass
        # Remove sufixos comuns: data e nome do veículo
        if article.title:
            article.title = re.sub(r"\s*[-–—]\s*\d{2}/\d{2}/\d{4}\s*$", '', article.title).strip()
            article.title = re.sub(r"\s*[-–—]\s*diário do grande abc\s*$", '', article.title, flags=re.IGNORECASE).strip()
            article.title = re.sub(r"\s*\|\s*diário do grande abc\s*$", '', article.title, flags=re.IGNORECASE).strip()
        # Remove lixo de índice/keywords que às vezes vira 'texto'
        if getattr(article, 'text', ''):
            txt = article.text or ''
            txt = re.sub(r"^\s*[-–—]\s*Notícias e informações do Grande ABC:.*$", '', txt, flags=re.IGNORECASE|re.MULTILINE)
            txt = re.sub(r"Notícias e informações do Grande ABC:.*", '', txt, flags=re.IGNORECASE)
            txt = re.sub(r"\n{3,}", '\n\n', txt).strip()
            article.text = txt
    # --- FIM LIMPEZA DGABC ---

    # V20: Filtrar índices
    if eh_pagina_indice(article.title or "", url):
        return None

    if not article.title:
        return None

    if verificar_relevancia:
        tem_termo, termo_achado = eh_relevante(article.title + " " + (article.text or ""), termos_monitorados)
        if not tem_termo:
            return None

    subtitulo = safe_str(article.meta_description)

    # DGABC: às vezes meta_description vira texto genérico/keywords (não é subtítulo)
    if ('dgabc' in get_domain(url)) and subtitulo and ('notícias e informações do grande abc' in subtitulo.lower()):
        subtitulo = ''
    if not subtitulo and article.text:
        p = article.text.split("\n")[0]
        if 10 < len(p) < 250:
            subtitulo = p

    permitidas = datas_permitidas()

    if data_rss:
        data_final = data_rss
    else:
        data_final = extrair_data_do_url(url)
        if not data_final and getattr(article, "publish_date", None):
            try:
                data_final = article.publish_date.strftime("%Y-%m-%d")
            except Exception:
                data_final = None

    if origem_manual != "Manual":
        if (not data_final) or (data_final not in permitidas):
            return None

    if origem_manual == "Manual" and not data_final:
        data_final = now_sp().strftime("%Y-%m-%d")

    if origem_manual == "Manual":
        nome_veiculo = identificar_fonte(url, lista_veiculos)
        if nome_veiculo == "Outros":
            nome_veiculo = "Importação Manual"
    else:
        nome_veiculo = identificar_fonte(url, lista_veiculos)

    legenda_padrao = f"Foto: Reprodução / {nome_veiculo}"
    dominio = get_domain(url)


    # Gazeta do Povo: não usar subtítulo
    if "gazetadopovo.com.br" in dominio:
        subtitulo = ""

    if nome_veiculo == "Agência SP":
        legenda_real = extrair_legenda_agenciasp(url)
        if legenda_real:
            legenda_padrao = legenda_real

    if ("gazetadopovo.com.br" in dominio) or (nome_veiculo == "Gazeta do Povo"):
        if html_bruto:
            ov = extrair_override_gazeta_do_povo(html_bruto)
            if ov.get("title"):
                article.title = ov["title"]
            if ov.get("imageCaption"):
                legenda_padrao = ov["imageCaption"]

    if ("poder360.com.br" in dominio) or (nome_veiculo == "Poder360"):
        if html_bruto:
            ov = extrair_override_poder360(html_bruto)
            if ov.get("title"):
                article.title = ov["title"]
            if ov.get("subtitle"):
                subtitulo = ov["subtitle"]
            if ov.get("author"):
                article.authors = [ov["author"]]
            if ov.get("imageCaption"):
                legenda_padrao = ov["imageCaption"]

    if ("folha.uol.com.br" in dominio) or (nome_veiculo == "Folha de S.Paulo"):
        if html_bruto:
            autor = extrair_autor_folha(html_bruto)
            if autor:
                autor = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", autor).strip()
                article.authors = [autor]

            soup = BeautifulSoup(html_bruto, "html.parser")
            ogt = soup.select_one("meta[property='og:title']")
            if ogt and (ogt.get("content") or "").strip():
                article.title = normalizar_titulo_folha(ogt.get("content") or "")

            leg = extrair_legenda_foto_folha(html_bruto)
            if leg:
                legenda_padrao = leg

    if ("g1.globo.com" in dominio) or (nome_veiculo == "G1"):
        if html_bruto:
            autor_g1 = extrair_autor_g1(html_bruto)
            if autor_g1:
                article.authors = [autor_g1]

            leg_g1 = extrair_legenda_foto_g1(html_bruto)
            if leg_g1:
                legenda_padrao = leg_g1


    # V20: ANÁLISE
    analise_sentimento = {"sentimento":"neutro","confianca":0.0,"justificativa":""}
    termo_usado = "Manual"
    try:
        if verificar_relevancia and "termo_achado" in locals(): termo_usado = termo_achado
        elif origem_manual: termo_usado = origem_manual
    except: pass
    if HABILITAR_SENTIMENTO and article.text:
        try:
            print(f"      🎭 Analisando...", flush=True)
            analise_sentimento = analisar_sentimento_groq(article.title+"\\n"+article.text[:1000], GROQ_API_KEY)
            e = obter_emoji_sentimento(analise_sentimento.get("sentimento","neutro"))
            s = analise_sentimento.get("sentimento","neutro")
            c = analise_sentimento.get("confianca",0.0)
            print(f"      {e} {s.upper()} ({c:.0%})", flush=True)
            if float(c) == 0.0 and (analise_sentimento.get("justificativa") or "").startswith("erro:"):
                print(f"      ⚠️ {analise_sentimento.get('justificativa')}", flush=True)
        except: pass

    return {
        "id": f"news-{hashlib.md5(url.encode()).hexdigest()[:10]}",
        "title": safe_str(article.title),
        "subtitle": safe_str(subtitulo),
        "author": safe_str(", ".join(article.authors)) if article.authors else "Redação",
        "date": data_final,
        "sources": [nome_veiculo],
        "columnists": [],
        "image": safe_str(article.top_image),
        "imageCaption": legenda_padrao,
        "imagePositionY": 50,
        "link": safe_str(url),
        "htmlContent": "",
        "fullStoryLink": safe_str(url),
        "showAccessDisclaimer": False,
        "politicalComment": "",
        "summaryParagraphs": 2,
        "commentParagraphs": 2,
        "themes": ["Fila de Entrada"],
        "content": safe_str(article.text).replace("\n", ""),
        "summary": "Pendente de Análise...",
        "ods": [],
        "sentimento": analise_sentimento.get("sentimento","neutro"),
        "sentimentoConfianca": analise_sentimento.get("confianca",0.0),
        "sentimentoJustificativa": analise_sentimento.get("justificativa",""),
        "sentimentoEmoji": obter_emoji_sentimento(analise_sentimento.get("sentimento","neutro")),
        "sentimentoCor": obter_cor_sentimento(analise_sentimento.get("sentimento","neutro")),
        "horaColeta": now_sp().strftime("%H:%M:%S"),
        "diaColeta": now_sp().strftime("%A"),
        "termoCapturado": termo_usado
    }


def extrair_noticia(
    url: str,
    lista_veiculos: List[str],
    termos_monitorados: List[str],
    origem_manual: Optional[str] = None,
    verificar_relevancia: bool = True,
    data_rss: Optional[str] = None
):
    if eh_url_bloqueada(url):
        return None

    for tentativa in range(1, MAX_TENTATIVAS_POR_ARTIGO + 1):
        try:
            print(f"         ⏬ [{tentativa}/{MAX_TENTATIVAS_POR_ARTIGO}] {url[:120]}", flush=True)
            resultado = _extrair_noticia_core(url, lista_veiculos, termos_monitorados, origem_manual, verificar_relevancia, data_rss)
            
            if resultado:
                print(f"         ✅ Capturada!", flush=True)
            
            return resultado
            
        except TimeoutError as e:
            print(f"         ⏱️ TIMEOUT na tentativa {tentativa}/{MAX_TENTATIVAS_POR_ARTIGO}: {e}", flush=True)
            if tentativa < MAX_TENTATIVAS_POR_ARTIGO:
                print(f"         🔄 Aguardando 2s antes de retentar...", flush=True)
                time.sleep(2)
            else:
                print(f"         ❌ DESISTINDO após {MAX_TENTATIVAS_POR_ARTIGO} tentativas", flush=True)
                return None
                
        except Exception as e:
            print(f"         ❌ Erro na tentativa {tentativa}/{MAX_TENTATIVAS_POR_ARTIGO}: {type(e).__name__}: {e}", flush=True)
            if tentativa < MAX_TENTATIVAS_POR_ARTIGO:
                time.sleep(1)
            else:
                return None
    
    return None



# ==============================================================================
# Salvamento parcial/incremental (robusto)
# ==============================================================================
def salvar_parcial(lista_noticias, colunistas_objs, lista_veiculos, temas_objs, arquivo=None):
    """Salva um snapshot parcial do estado atual das notícias.

    Usado em:
      - auto-save periódico
      - Ctrl+C (signal_handler)
      - recuperação após erro inesperado

    Observação: escreve em arquivo temporário e faz rename atômico para reduzir
    chance de arquivo JSON corrompido.
    """
    estrutura_final = {
        "news": lista_noticias,
        "columnists": colunistas_objs,
        "monitoredVehicles": lista_veiculos,
        "themes": temas_objs,
        "sources": lista_veiculos,
    }

    try:
        if arquivo is None:
            arquivo = str(BASE_DIR / "fila_para_curadoria_PARCIAL.json")

        tmp = arquivo + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(estrutura_final, f, ensure_ascii=False, indent=2)
        os.replace(tmp, arquivo)

        print(f"💾 [AUTO-SAVE] {len(lista_noticias)} notícias salvas em: {arquivo}", flush=True)
    except Exception as e:
        print(f"❌ Erro ao salvar parcial: {e}")

# ==============================================================================
# --- MOTOR ---
# ==============================================================================
def rodar_motor_v26():
    print("\n🏭 --- MOTOR V26 (Gazeta sem subtítulo + filtros) ---")
    if HABILITAR_SENTIMENTO: print("✅ Análise HABILITADA")
    else: print("⚠️ Análise DESABILITADA")
    print("📅 Datas permitidas:", ", ".join(sorted(datas_permitidas())))

    try:
        fontes, termos_monitorados, lista_colunistas, lista_temas, lista_veiculos = carregar_configuracoes()
        print(f"🧩 Config carregada: {len(fontes)} fontes, {len(termos_monitorados)} termos, "
              f"{len(lista_colunistas)} colunistas, {len(lista_temas)} temas, {len(lista_veiculos)} veículos.")
    except Exception as e:
        print(f"\n❌ Erro ao carregar TXT de configuração: {e}")
        print("➡️ Verifique se os arquivos .txt existem no mesmo diretório do motor e estão no formato correto.")
        return

    colunistas_objs = converter_colunistas(lista_colunistas)
    temas_objs = converter_temas(lista_temas)

    lista_noticias: List[Dict] = []
    urls_ja_lidas = set()
    # V26: Carregar URLs já processadas em execuções anteriores
    urls_processadas = carregar_urls_processadas()
    urls_ja_lidas.update(urls_processadas)
    urls_tentadas = carregar_urls_tentadas()
    urls_ja_lidas.update(urls_tentadas)
    limpar_urls_antigas()  # Auto-limpeza se necessário

    contador_desde_ultimo_save = 0

    def signal_handler(sig, frame):
        print("\n\n⚠️ Interrupção detectada (Ctrl+C). Salvando progresso parcial...")
        salvar_parcial(lista_noticias, colunistas_objs, lista_veiculos, temas_objs)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 1) LINKS MANUAIS
    if os.path.exists(ARQUIVO_LINKS_MANUAIS):
        print("📋 Links Manuais...")
        with open(ARQUIVO_LINKS_MANUAIS, "r", encoding="utf-8") as f:
            links = [l.strip() for l in f if l.strip()]

        for link in links:
            if link in urls_ja_lidas:
                continue

            print(f"   -> Link: {link[:90]}...")
            salvar_url_tentada(link)
            urls_ja_lidas.add(link)
            item = extrair_noticia(
                link,
                lista_veiculos=lista_veiculos,
                termos_monitorados=termos_monitorados,
                origem_manual="Manual",
                verificar_relevancia=False
            )
            if item:
                lista_noticias.append(item)
                urls_ja_lidas.add(link)
                salvar_url_processada(link)
                contador_desde_ultimo_save += 1
                
                if contador_desde_ultimo_save >= SALVAR_A_CADA:
                    salvar_parcial(lista_noticias, colunistas_objs, lista_veiculos, temas_objs)
                    contador_desde_ultimo_save = 0
    else:
        open(ARQUIVO_LINKS_MANUAIS, "w", encoding="utf-8").close()

    # 2) VARREDURA FONTES (RSS + HTML)
    print("\n📡 Iniciando Varredura Automática...")
    permitidas = datas_permitidas()

    for idx_fonte, fonte in enumerate(fontes, start=1):
        nome_canal = fonte["nome"]
        url_origem = fonte["url"]
        tipo = fonte["tipo"]
        if PULAR_GAZETA_HTML and ('Gazeta do Povo' in nome_canal) and (tipo == 'HTML'):
            print('      ⚠️ Pulando Gazeta do Povo (HTML) por instabilidade no Windows.', flush=True)
            continue

        print(f"\n   🌍 [{idx_fonte}/{len(fontes)}] {nome_canal} ({tipo})", flush=True)

        if tipo == "RSS":
            rss_bytes = baixar_rss_bytes(url_origem, tentativas=2)
            if not rss_bytes:
                continue

            try:
                feed = feedparser.parse(rss_bytes)
                if getattr(feed, "bozo", 0):
                    exc = getattr(feed, "bozo_exception", None)
                    if exc:
                        print(f"      ⚠️ RSS com problema (bozo): {exc}")

                entries = getattr(feed, "entries", []) or []
                print(f"      🔗 {len(entries)} notícias.", flush=True)

                capturadas = 0
                for entry in entries:
                    if capturadas >= MAX_NOTICIAS_POR_FONTE:
                        break

                    link = getattr(entry, "link", None)
                    if not link:
                        continue
                    if link in urls_ja_lidas:
                        continue

                    data_rss_str = extrair_data_rss(entry)
                    if data_rss_str and (data_rss_str not in permitidas):
                        continue

                    titulo = safe_str(getattr(entry, "title", ""))
                    desc = safe_str(getattr(entry, "description", "")) or safe_str(entry.get("summary", ""))

                    tem_termo, _ = eh_relevante(titulo + " " + desc, termos_monitorados)
                    if not tem_termo:
                        continue

                    salvar_url_tentada(link)
                    urls_ja_lidas.add(link)
                    item = extrair_noticia(
                        link,
                        lista_veiculos=lista_veiculos,
                        termos_monitorados=termos_monitorados,
                        origem_manual="Auto",
                        verificar_relevancia=True,
                        data_rss=data_rss_str
                    )
                    if item:
                        lista_noticias.append(item)
                        urls_ja_lidas.add(link)
                        salvar_url_processada(link)
                        capturadas += 1
                        contador_desde_ultimo_save += 1
                        
                        if contador_desde_ultimo_save >= SALVAR_A_CADA:
                            salvar_parcial(lista_noticias, colunistas_objs, lista_veiculos, temas_objs)
                            contador_desde_ultimo_save = 0

                print(f"      ✔️ {capturadas} capturadas de {nome_canal}", flush=True)

            except Exception as e:
                print(f"      ❌ Erro ao processar RSS: {e}", flush=True)

        else:
            try:
                links = extrair_links_de_secao(url_origem, limite=220)
                print(f"      🔗 {len(links)} links (candidatos a matéria) após filtro.")
            except Exception as e:
                print(f"      ❌ Erro ao processar seção HTML: {e}")
                continue

            capturadas = 0
            for link in links:
                if capturadas >= MAX_NOTICIAS_POR_FONTE:
                    break
                if link in urls_ja_lidas:
                    continue

                durl = extrair_data_do_url(link)
                if durl and (durl not in permitidas):
                    continue

                salvar_url_tentada(link)
                urls_ja_lidas.add(link)
                item = extrair_noticia(
                    link,
                    lista_veiculos=lista_veiculos,
                    termos_monitorados=termos_monitorados,
                    origem_manual="Auto",
                    verificar_relevancia=True,
                    data_rss=None
                )
                if item:
                    lista_noticias.append(item)
                    urls_ja_lidas.add(link)
                    salvar_url_processada(link)
                    capturadas += 1
                    contador_desde_ultimo_save += 1
                    
                    if contador_desde_ultimo_save >= SALVAR_A_CADA:
                        salvar_parcial(lista_noticias, colunistas_objs, lista_veiculos, temas_objs)
                        contador_desde_ultimo_save = 0

            print(f"      ✔️ {capturadas} capturadas de {nome_canal}", flush=True)

    # 3) SALVAR FINAL
    estrutura_final = {
        "news": lista_noticias,
        "columnists": colunistas_objs,
        "monitoredVehicles": lista_veiculos,
        "themes": temas_objs,
        "sources": lista_veiculos
    }

    try:
        with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
            json.dump(estrutura_final, f, ensure_ascii=False, indent=2)
        print(f"\n✅ SUCESSO! {len(lista_noticias)} notícias capturadas.")
        print(f"📂 Arquivo gerado: {ARQUIVO_SAIDA}")
    except Exception as e:
        print(f"\n❌ Erro ao salvar: {e}")



if __name__ == "__main__":
    """Loop supervisor: tenta rodar o motor e, em caso de erro inesperado,
    aguarda alguns segundos e tenta novamente.

    O motor já carrega urls_processadas.txt no início e pula URLs antigas.
    """
    import traceback

    while True:
        try:
            rodar_motor_v26()
            print("")
            print("✅ Execução concluída com sucesso. Encerrando.")
            break
        except KeyboardInterrupt:
            print("")
            print("⚠️ Interrompido manualmente (Ctrl+C). Encerrando sem reinício automático.")
            break
        except Exception as e:
            print("")
            print("❌ ERRO INESPERADO NO MOTOR:")
            print(f"   Tipo: {type(e).__name__}")
            print(f"   Detalhe: {e}")
            traceback.print_exc()

            try:
                salvar_parcial(
                    globals().get('lista_noticias', []),
                    globals().get('colunistas_objs', []),
                    globals().get('lista_veiculos', []),
                    globals().get('temas_objs', []),
                )
            except Exception:
                pass

            print("")
            print("⏳ Aguardando 10 segundos antes de tentar continuar de onde parou...")
            time.sleep(10)
