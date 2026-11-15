import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib

import preprocessamento  # noqa: F401  (necessário para carregar o pipeline salvo)

PASTA = Path(__file__).resolve().parent
ARQ_MODELO = PASTA / "modelos" / "modelo_final.joblib"
ARQ_META = PASTA / "modelos" / "metadados.json"

LIMIAR = 0.5            # probabilidade a partir da qual a notícia é marcada como potencialmente falsa
FAIXA_INCERTA = 0.15    # entre 0,35 e 0,65 o resultado é apresentado como inconclusivo
MIN_PALAVRAS = 50       # textos curtos demais tornam a estimativa pouco confiável
N_TERMOS = 8


# ---------------------------------------------------------------------------
# Lógica (independente da interface)
# ---------------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(ARQ_MODELO)
    meta = json.loads(ARQ_META.read_text(encoding="utf-8")) if ARQ_META.exists() else {}
    return modelo, meta


def analisar(modelo, texto, n_termos=N_TERMOS):
    """Retorna a probabilidade de fake e, se o classificador for linear,
    os termos que mais empurraram a estimativa para cada classe."""
    prob_fake = float(modelo.predict_proba([texto])[0][1])

    termos_fake, termos_true = [], []
    clf = modelo.named_steps["clf"]
    if hasattr(clf, "coef_"):
        limpo = modelo.named_steps["prep"].transform([texto])
        x = modelo.named_steps["tfidf"].transform(limpo)
        vocab = modelo.named_steps["tfidf"].get_feature_names_out()
        contrib = x.multiply(clf.coef_[0]).tocsr()          # peso TF-IDF × coeficiente
        idx, val = contrib.indices, contrib.data
        ordem = np.argsort(val)
        termos_fake = [(vocab[idx[i]], val[i]) for i in ordem[::-1][:n_termos] if val[i] > 0]
        termos_true = [(vocab[idx[i]], -val[i]) for i in ordem[:n_termos] if val[i] < 0]
    return prob_fake, termos_fake, termos_true


def classificar(prob_fake):
    if abs(prob_fake - LIMIAR) < FAIXA_INCERTA:
        return "inconclusivo"
    return "fake" if prob_fake >= LIMIAR else "true"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Identificador de Fake News", page_icon="📰", layout="centered")

st.title("Identificador de Fake News")
st.caption("Protótipo acadêmico – Inteligência Artificial, Universidade Presbiteriana Mackenzie")

st.info(
    "Esta ferramenta é um **apoio à análise**, não um verificador de fatos. Ela estima, a partir de "
    "padrões de escrita aprendidos em notícias em português de 2016 a 2018, se o texto se parece "
    "mais com notícias falsas ou verdadeiras. **Não confirma a veracidade de nenhuma informação.** "
    "Na dúvida, consulte agências de checagem e fontes oficiais."
)

if not ARQ_MODELO.exists():
    st.error(f"Modelo não encontrado em `{ARQ_MODELO}`. Rode o notebook 02_modelagem.ipynb antes.")
    st.stop()

modelo, meta = carregar_modelo()

titulo = st.text_input("Título da notícia:", placeholder="Ex.: Governo anuncia novo programa de ...")
corpo = st.text_area("Corpo da notícia (texto completo, sem o título):", height=260,
                     placeholder="Cole aqui o texto da notícia...")
analisar_btn = st.button("Analisar", type="primary", use_container_width=True)

if analisar_btn:
    if not corpo.strip():
        st.warning("Cole o corpo da notícia para analisar.")
        st.stop()
    if not titulo.strip():
        st.caption("Sem título: a análise usará apenas o corpo. Incluir o título tende a deixar a "
                   "estimativa mais parecida com as notícias usadas no treinamento.")
    # No Fake.Br Corpus o título é a primeira linha do texto, seguido do corpo
    texto = f"{titulo.strip()}\n{corpo.strip()}".strip()
    n_palavras = len(texto.split())
    if n_palavras < MIN_PALAVRAS:
        st.warning(f"O texto tem apenas {n_palavras} palavras. Com textos curtos a estimativa é "
                   f"pouco confiável; o ideal é colar a notícia inteira (pelo menos {MIN_PALAVRAS} palavras).")

    prob_fake, termos_fake, termos_true = analisar(modelo, texto)
    situacao = classificar(prob_fake)

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    col1.metric("Probabilidade estimada de ser falsa", f"{prob_fake:.0%}")
    col2.metric("Probabilidade estimada de ser verdadeira", f"{1 - prob_fake:.0%}")
    st.progress(prob_fake)

    if situacao == "fake":
        st.warning("**Potencialmente falsa.** O texto apresenta padrões mais parecidos com os das "
                   "notícias falsas do conjunto de treinamento. Verifique a informação em outras fontes "
                   "antes de compartilhar.")
    elif situacao == "true":
        st.success("**Potencialmente verdadeira.** O texto apresenta padrões mais parecidos com os das "
                   "notícias verdadeiras do conjunto de treinamento. Isso não garante que o conteúdo "
                   "seja verdadeiro.")
    else:
        st.info("**Inconclusivo.** A probabilidade ficou próxima de 50%; o modelo não encontrou "
                "padrões suficientes para indicar uma das classes.")

    if termos_fake or termos_true:
        with st.expander("Quais palavras mais influenciaram a estimativa?"):
            st.caption("Termos do texto (após a limpeza) com maior peso na decisão do modelo. "
                       "Indicam padrões estatísticos, não erros factuais.")
            c1, c2 = st.columns(2)
            c1.markdown("**Aproximaram de _falsa_**")
            c1.markdown("\n".join(f"- {t}" for t, _ in termos_fake) or "_nenhum_")
            c2.markdown("**Aproximaram de _verdadeira_**")
            c2.markdown("\n".join(f"- {t}" for t, _ in termos_true) or "_nenhum_")

with st.expander("Sobre o modelo e suas limitações"):
    f1 = meta.get("cv_f1_macro")
    st.markdown(
        f"- **Modelo:** {meta.get('modelo', 'n/d')} sobre representação TF-IDF "
        f"({meta.get('texto', 'n/d')}).\n"
        f"- **Dados:** Fake.Br Corpus (versão {meta.get('versao_corpus', 'n/d')}), notícias em português "
        "publicadas entre 2016 e 2018, em seis categorias (política, TV e celebridades, sociedade e "
        "cotidiano, ciência e tecnologia, economia e religião).\n"
        + (f"- **Desempenho:** F1-score macro de {f1:.1%} na validação cruzada. Ou seja, o modelo "
           "também erra: notícias verdadeiras podem ser marcadas como falsas e vice-versa.\n" if f1 else "")
        + "- **Limitações:** o modelo analisa padrões de vocabulário, não checa fatos. Pode errar com "
        "temas, estilos ou acontecimentos posteriores a 2018, com textos curtos, e com notícias "
        "satíricas ou de opinião.\n"
        "- **Privacidade:** o texto digitado é processado apenas na memória e não é armazenado."
    )