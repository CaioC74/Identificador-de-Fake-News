import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

_RE_CP1252 = re.compile("[\x80-\x9f]")


def corrigir_codificacao(texto):
    """Remove o BOM (\\ufeff) e converte caracteres de controle herdados do
    Windows-1252 (ex.: \\x93 \\x94 -> aspas curvas) para os caracteres corretos."""
    texto = texto.replace("\ufeff", "")
    return _RE_CP1252.sub(lambda m: m.group().encode("latin-1").decode("cp1252", errors="ignore"), texto)

def carregar_versao(corpus_dir, versao="size_normalized_texts"):
    """Lê <corpus_dir>/<versao>/fake/*.txt e <corpus_dir>/<versao>/true/*.txt.

    No Fake.Br, o arquivo N.txt de fake/ e o arquivo N.txt de true/ formam um par
    (mesmo assunto). O número do arquivo é usado como identificador do par.
    Rótulo: 1 = fake, 0 = true.
    """
    base = Path(corpus_dir) / versao
    registros = []
    for classe, rotulo in (("fake", 1), ("true", 0)):
        pasta = base / classe
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
        for arq in pasta.glob("*.txt"):
            if not arq.stem.isdigit():
                continue
            texto = corrigir_codificacao(arq.read_text(encoding="utf-8", errors="replace"))
            registros.append({
                "par_id": int(arq.stem),
                "classe": classe,
                "rotulo": rotulo,
                "texto": texto,
            })
    df = pd.DataFrame(registros).sort_values(["par_id", "rotulo"]).reset_index(drop=True)
    df["n_palavras"] = df["texto"].str.split().str.len()
    return df


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

_RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
_RE_HTML = re.compile(r"<.*?>")
_RE_TOKEN = re.compile(r"[^\W\d_]+")  # sequências de letras (inclui acentos); descarta números e pontuação

_stopwords = None
_stemmer = None


def _get_stopwords():
    global _stopwords
    if _stopwords is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words("portuguese"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words("portuguese"))
    return _stopwords


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        import nltk
        try:
            _stemmer = nltk.stem.RSLPStemmer()
        except LookupError:
            nltk.download("rslp", quiet=True)
            _stemmer = nltk.stem.RSLPStemmer()
    return _stemmer


def limpar_texto(texto, radicalizar=False):
    """Minúsculas, remove URLs, tags HTML, números, pontuação, palavras de 1 caractere
    e stopwords; opcionalmente aplica o stemmer RSLP."""
    sw = _get_stopwords()
    texto = _RE_HTML.sub(" ", _RE_URL.sub(" ", texto.lower()))
    tokens = [t for t in _RE_TOKEN.findall(texto) if len(t) > 1 and t not in sw]
    if radicalizar:
        tokens = [_radical(t) for t in tokens]
    return " ".join(tokens)


@lru_cache(maxsize=None)
def _radical(token):
    # cache: o RSLP é lento e as mesmas palavras se repetem muito no corpus
    return _get_stemmer().stem(token)


def limpar_lote(textos, radicalizar=False):
    """Versão para listas/Series – usada no FunctionTransformer do pipeline final."""
    return [limpar_texto(t, radicalizar=radicalizar) for t in textos]


def limpar_lote_radicalizado(textos):
    return limpar_lote(textos, radicalizar=True)