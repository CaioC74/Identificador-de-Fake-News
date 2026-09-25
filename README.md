# Identificador-de-Fake-News

Projeto desenvolvido para a disciplina de **Inteligência Artificial** da Universidade Presbiteriana Mackenzie.

O projeto tem como objetivo desenvolver um modelo de **Aprendizado de Máquina** capaz de analisar textos de notícias em português brasileiro e estimar a qual classe o texto está associado, com base em padrões identificados no conjunto de dados utilizado.

**Importante:** o sistema não realiza uma verificação definitiva da veracidade de uma notícia. A classificação representa uma estimativa baseada nos padrões aprendidos pelo modelo a partir dos dados utilizados no treinamento.

## Objetivos

* Aplicar técnicas de **Inteligência Artificial** em um problema relacionado à disseminação de desinformação;
* Utilizar **Processamento de Linguagem Natural (PLN)** para trabalhar com textos de notícias;
* Aplicar técnicas de **Aprendizado de Máquina supervisionado** para classificação de textos;
* Comparar diferentes algoritmos de classificação;
* Avaliar o desempenho dos modelos utilizando métricas de classificação;
* Desenvolver um protótipo para demonstrar a aplicação do modelo.

## Dataset

O projeto utiliza o **Fake.Br Corpus**, um conjunto de dados composto por notícias em português brasileiro classificadas como `fake` ou `true`.

O conjunto de dados utilizado possui:

* **7.200 notícias**;
* **3.600 notícias falsas**;
* **3.600 notícias verdadeiras**;
* Textos em **português brasileiro**;
* Notícias coletadas entre **2016 e 2018**;
* Categorias como política, economia, religião, ciência e tecnologia, sociedade e cotidiano e TV e celebridades.

A base de dados não está armazenada neste repositório. Para reproduzir as análises e os experimentos, o dataset deve ser obtido diretamente em seu repositório de origem:

**Fake.Br Corpus:**
https://github.com/roneysco/Fake.Br-Corpus

O dataset é utilizado no projeto para as etapas de análise exploratória, preparação dos textos e treinamento dos modelos de classificação.

Monteiro R.A., Santos R.L.S., Pardo T.A.S., de Almeida T.A., Ruiz E.E.S., Vale O.A. (2018) Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. In: Villavicencio A. et al. (eds) Computational Processing of the Portuguese Language. PROPOR 2018. Lecture Notes in Computer Science, vol 11122. Springer, Cham

## Tecnologias

O projeto utiliza:

* **Python**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

## Técnicas utilizadas

As principais técnicas utilizadas ou previstas no projeto são:

* **Processamento de Linguagem Natural (PLN)**
* **TF-IDF**
* **Aprendizado de Máquina supervisionado**
* **Classificação de textos**

## Passos Seguidos

### 1. Análise exploratória

Inicialmente, são analisadas características do conjunto de dados, incluindo:

* distribuição entre as classes `true` e `fake`;
* distribuição das notícias por categoria;
* distribuição das categorias entre as classes;
* quantidade média de palavras por classe;
* palavras mais frequentes no conjunto de dados;
* palavras mais frequentes nas notícias `true`;
* palavras mais frequentes nas notícias `fake`;
* valores ausentes;
* textos duplicados.

Os gráficos gerados pela análise exploratória são armazenados na pasta `graficos/`.

### 2. Preparação dos textos

Os textos serão preparados para utilização nos modelos de Aprendizado de Máquina. Essa etapa inclui procedimentos de limpeza e transformação dos textos e sua representação utilizando **TF-IDF**.

### 3. Treinamento

Após a preparação dos dados, serão treinados diferentes modelos de classificação para comparar seu desempenho na identificação das classes `true` e `fake`.

### 4. Avaliação

Os modelos serão avaliados utilizando métricas de classificação.

## Estrutura do projeto

```text
Fake.Br-Corpus/
│
├── Documentacao/
│   ├── Artigo Parcial.pdf
│   │
│   └── graficos/
│       ├── 01_distribuicao_classes.png
│       ├── 02_distribuicao_categorias.png
│       ├── 03_categorias_por_classe.png
│       ├── 04_media_palavras.png
│       ├── 05_palavras_mais_frequentes.png
│       ├── 06_palavras_true.png
│       └── 07_palavras_fake.png
│
├── analise_exploratoria.py
├── README.md
```

## Resultados

Os resultados serão adicionados ao repositório conforme o desenvolvimento do projeto.

Entre os resultados previstos estão:

* análise exploratória do dataset;
* comparação dos modelos de classificação;
* métricas de avaliação;
* matrizes de confusão;
* análise dos resultados obtidos;
* protótipo para classificação de novos textos.

## ODS

O projeto está relacionado ao **ODS 16 – Paz, Justiça e Instituições Eficazes**, especialmente à **Meta 16.10**, relacionada ao acesso público à informação e à proteção das liberdades fundamentais.

## Limitações

O sistema não deve ser utilizado como um mecanismo definitivo de verificação de fatos.

A classificação realizada pelo modelo representa uma **estimativa baseada nos padrões presentes nos dados utilizados para seu treinamento**. Dessa forma, um texto classificado como `fake` não significa necessariamente que seu conteúdo seja falso, assim como uma classificação `true` não garante a veracidade das informações apresentadas.

## Status do projeto

**Em desenvolvimento**

Atualmente, o projeto possui a etapa de **análise exploratória do conjunto de dados** implementada. As próximas etapas envolvem a preparação dos textos, aplicação do TF-IDF, treinamento dos modelos, avaliação dos resultados e desenvolvimento do protótipo.

## Autores

Projeto desenvolvido por estudantes do curso de **Ciência da Computação — Universidade Presbiteriana Mackenzie**.
* Caio Cesar Navarro Pugliese
* Enzo Maranho Tucilho
