import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

BASE = "full_texts"

PASTA_FAKE = os.path.join(BASE, "fake")
PASTA_TRUE = os.path.join(BASE, "true")

PASTA_META_FAKE = os.path.join(BASE, "fake-meta-information")
PASTA_META_TRUE = os.path.join(BASE, "true-meta-information")

PASTA_GRAFICOS = "graficos"

# Caso o código esteja sendo executado diretamente a partir do repositório do GitHub, alterar o caminho para:
# PASTA_GRAFICOS = os.path.join("Documentacao", "graficos")

os.makedirs(PASTA_GRAFICOS, exist_ok=True)


# ============================================================
# 2 - LEITURA DOS TEXTOS
# ============================================================

dados = []

# Notícias fake
for arquivo in os.listdir(PASTA_FAKE):

    caminho = os.path.join(PASTA_FAKE, arquivo)

    if os.path.isfile(caminho):

        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        dados.append({
            "arquivo": arquivo,
            "texto": texto,
            "classe": "fake",
            "qtd_palavras": len(texto.split())
        })


# Notícias true
for arquivo in os.listdir(PASTA_TRUE):

    caminho = os.path.join(PASTA_TRUE, arquivo)

    if os.path.isfile(caminho):

        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        dados.append({
            "arquivo": arquivo,
            "texto": texto,
            "classe": "true",
            "qtd_palavras": len(texto.split())
        })


df = pd.DataFrame(dados)


# ============================================================
# 3 - DISTRIBUIÇÃO DAS CLASSES
# ============================================================

print("\n========================================")
print("1 - DISTRIBUIÇÃO DAS CLASSES")
print("========================================")

distribuicao = df["classe"].value_counts()

print(distribuicao)


plt.figure(figsize=(8, 5))

distribuicao.plot(kind="bar")

plt.title("Distribuição das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade de notícias")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "01_distribuicao_classes.png"),
    dpi=300
)

plt.close()


# ============================================================
# 4 - LEITURA DAS CATEGORIAS
# ============================================================

categorias_conhecidas = {
    "politica",
    "tv_celebridades",
    "sociedade_cotidiana",
    "ciencia_tecnologia",
    "economia",
    "religiao"
}

categorias = []


def ler_categorias(pasta_meta, classe):

    resultado = []

    for arquivo in os.listdir(pasta_meta):

        caminho = os.path.join(pasta_meta, arquivo)

        if not os.path.isfile(caminho):
            continue

        try:

            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                linhas = [linha.strip() for linha in f.readlines()]

            if len(linhas) >= 3:

                categoria = linhas[2]

                if categoria in categorias_conhecidas:

                    resultado.append({
                        "arquivo": arquivo,
                        "categoria": categoria,
                        "classe": classe
                    })

        except Exception:
            pass

    return resultado


categorias.extend(
    ler_categorias(PASTA_META_FAKE, "fake")
)

categorias.extend(
    ler_categorias(PASTA_META_TRUE, "true")
)


df_categorias = pd.DataFrame(categorias)


# ============================================================
# 5 - DISTRIBUIÇÃO POR CATEGORIA
# ============================================================

print("\n========================================")
print("2 - DISTRIBUIÇÃO POR CATEGORIA")
print("========================================")

distribuicao_categorias = df_categorias["categoria"].value_counts()

print(distribuicao_categorias)


plt.figure(figsize=(10, 6))

distribuicao_categorias.plot(kind="bar")

plt.title("Distribuição das notícias por categoria")
plt.xlabel("Categoria")
plt.ylabel("Quantidade de notícias")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "02_distribuicao_categorias.png"),
    dpi=300
)

plt.close()


# ============================================================
# 6 - CATEGORIAS POR CLASSE
# ============================================================

print("\n========================================")
print("3 - CATEGORIAS POR CLASSE")
print("========================================")

categorias_classe = pd.crosstab(
    df_categorias["categoria"],
    df_categorias["classe"]
)

print(categorias_classe)


categorias_classe.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Distribuição das categorias por classe")
plt.xlabel("Categoria")
plt.ylabel("Quantidade de notícias")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "03_categorias_por_classe.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7 - MÉDIA DE PALAVRAS POR CLASSE
# ============================================================

print("\n========================================")
print("4 - MÉDIA DE PALAVRAS")
print("========================================")

media_palavras = df.groupby("classe")["qtd_palavras"].mean()

print(media_palavras)


plt.figure(figsize=(8, 5))

media_palavras.plot(kind="bar")

plt.title("Média de palavras por classe")
plt.xlabel("Classe")
plt.ylabel("Média de palavras")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "04_media_palavras.png"),
    dpi=300
)

plt.close()


# ============================================================
# 8 - VALORES AUSENTES
# ============================================================

print("\n========================================")
print("5 - VALORES AUSENTES")
print("========================================")

print(
    df[
        ["arquivo", "texto", "classe", "qtd_palavras"]
    ].isnull().sum()
)


# ============================================================
# 9 - DUPLICATAS
# ============================================================

print("\n========================================")
print("6 - DUPLICATAS")
print("========================================")

duplicatas = df["texto"].duplicated().sum()

print("Quantidade de textos duplicados:", duplicatas)


# ============================================================
# 10 - PREPARAÇÃO DAS PALAVRAS
# ============================================================

stopwords = {
    "a", "o", "e", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas",
    "para", "por", "com", "sem",
    "que", "se", "ao", "aos", "às", "à",
    "os", "as",
    "é", "foi", "ser", "são",
    "como", "mais", "já", "também",
    "ou", "mas", "não", "sim",
    "seu", "sua", "seus", "suas",
    "ele", "ela", "eles", "elas",
    "isso", "essa", "esse", "este", "esta",
    "sobre", "entre", "até",
    "pela", "pelo", "pelas", "pelos",
    "ter", "tem", "têm",
    "muito", "muita", "muitos", "muitas"
}


def limpar_texto(texto):

    # Remove URLs
    texto = re.sub(r"http\S+|www\S+", " ", texto)

    # Remove HTML
    texto = re.sub(r"<.*?>", " ", texto)

    # Mantém apenas letras
    texto = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", texto)

    # Converte para minúsculas
    texto = texto.lower()

    # Divide em palavras
    palavras = texto.split()

    # Remove stopwords e palavras muito curtas
    palavras = [
        palavra
        for palavra in palavras
        if palavra not in stopwords
        and len(palavra) > 1
    ]

    return palavras


# ============================================================
# 11 - PALAVRAS MAIS FREQUENTES - GERAL
# ============================================================

print("\n========================================")
print("7 - PALAVRAS MAIS FREQUENTES")
print("========================================")

todas_palavras = []

for texto in df["texto"]:

    palavras = limpar_texto(texto)

    todas_palavras.extend(palavras)


contador = Counter(todas_palavras)

mais_frequentes = contador.most_common(20)

print(mais_frequentes)


palavras_df = pd.DataFrame(
    mais_frequentes,
    columns=["palavra", "frequencia"]
)


plt.figure(figsize=(10, 6))

plt.bar(
    palavras_df["palavra"],
    palavras_df["frequencia"]
)

plt.title("20 palavras mais frequentes")
plt.xlabel("Palavra")
plt.ylabel("Frequência")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "05_palavras_mais_frequentes.png"),
    dpi=300
)

plt.close()


# ============================================================
# 12 - PALAVRAS MAIS FREQUENTES NAS NOTÍCIAS TRUE
# ============================================================

print("\n========================================")
print("8 - PALAVRAS MAIS FREQUENTES - TRUE")
print("========================================")

palavras_true = []

for texto in df[df["classe"] == "true"]["texto"]:

    palavras = limpar_texto(texto)

    palavras_true.extend(palavras)


contador_true = Counter(palavras_true)

mais_frequentes_true = contador_true.most_common(20)

print(mais_frequentes_true)


df_true = pd.DataFrame(
    mais_frequentes_true,
    columns=["palavra", "frequencia"]
)


plt.figure(figsize=(10, 6))

plt.bar(
    df_true["palavra"],
    df_true["frequencia"]
)

plt.title("20 palavras mais frequentes nas notícias verdadeiras")
plt.xlabel("Palavra")
plt.ylabel("Frequência")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "06_palavras_true.png"),
    dpi=300
)

plt.close()


# ============================================================
# 13 - PALAVRAS MAIS FREQUENTES NAS NOTÍCIAS FAKE
# ============================================================

print("\n========================================")
print("9 - PALAVRAS MAIS FREQUENTES - FAKE")
print("========================================")

palavras_fake = []

for texto in df[df["classe"] == "fake"]["texto"]:

    palavras = limpar_texto(texto)

    palavras_fake.extend(palavras)


contador_fake = Counter(palavras_fake)

mais_frequentes_fake = contador_fake.most_common(20)

print(mais_frequentes_fake)


df_fake = pd.DataFrame(
    mais_frequentes_fake,
    columns=["palavra", "frequencia"]
)


plt.figure(figsize=(10, 6))

plt.bar(
    df_fake["palavra"],
    df_fake["frequencia"]
)

plt.title("20 palavras mais frequentes nas notícias falsas")
plt.xlabel("Palavra")
plt.ylabel("Frequência")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(PASTA_GRAFICOS, "07_palavras_fake.png"),
    dpi=300
)

plt.close()


# ============================================================
# 14 - RESUMO
# ============================================================

print("\n========================================")
print("ANÁLISE CONCLUÍDA")
print("========================================")

print("Total de notícias:", len(df))

print(
    "Notícias fake:",
    len(df[df["classe"] == "fake"])
)

print(
    "Notícias true:",
    len(df[df["classe"] == "true"])
)

print(
    "Textos duplicados:",
    duplicatas
)

print("\nGráficos salvos na pasta:", PASTA_GRAFICOS)
