# Identificador de Fake News

Projeto da disciplina de **Inteligência Artificial** (7º semestre – Ciência da Computação) da **Universidade Presbiteriana Mackenzie**.

Aplicação de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina que recebe o texto de uma notícia em português e estima se ela está mais associada a conteúdos **potencialmente falsos** ou **potencialmente verdadeiros**.

> ⚠️ **Aviso:** a ferramenta é um **apoio à análise** e **não verifica a veracidade** das notícias. O resultado é uma estimativa baseada em padrões de escrita aprendidos com o corpus de treinamento e pode conter erros.

**Integrantes**
- Caio Cesar Navarro Pugliese (RA 10436413)
- Enzo Maranho Tucilho (RA 10436106)

**ODS relacionado:** ODS 16 – Paz, Justiça e Instituições Eficazes (meta 16.10 – acesso público à informação).

🎥 **Vídeo de apresentação**

[![Vídeo de apresentação do projeto](https://img.youtube.com/vi/ID_DO_VIDEO/0.jpg)]([https://youtu.be/ID_DO_VIDEO](https://youtu.be/X-8_TtTatag))

---

## Sumário
1. [Dataset](#dataset)
2. [Metodologia](#metodologia)
3. [Resultados](#resultados)
4. [Estrutura do repositório](#estrutura-do-repositório)
5. [Como executar](#como-executar)
6. [Uso de inteligência artificial](#uso-de-inteligência-artificial)
7. [Referências](#referências)

---

## Dataset

**Fake.Br Corpus** (Monteiro et al., 2018): 7.200 notícias em português brasileiro (3.600 falsas e 3.600 verdadeiras), coletadas entre 2016 e 2018, nas categorias política, TV e celebridades, sociedade e cotidiano, ciência e tecnologia, economia e religião. Cada notícia falsa tem uma notícia verdadeira correspondente sobre o mesmo assunto (pares alinhados).

O corpus **não está neste repositório**. Ele deve ser baixado do repositório oficial: <https://github.com/roneysco/Fake.br-Corpus>

São usadas duas versões:
- `size_normalized_texts`: cada par de notícias tem tamanho semelhante. **Versão principal.**
- `full_texts`: textos completos. Usada para comparação, porque nela as notícias verdadeiras são cerca de 6 vezes mais longas que as falsas.

## Metodologia

1. **Análise exploratória:** distribuição de classes e categorias, comprimento dos textos, valores ausentes, duplicatas e palavras mais frequentes.
2. **Preparação:** remoção do par com notícia duplicada, uso apenas dos pares presentes nas duas versões, correção de codificação e limpeza do texto (minúsculas, remoção de URLs, HTML, números, pontuação e stopwords).
3. **Divisão por pares:** 80% treino / 20% teste, mantendo as duas notícias de um mesmo par no mesmo conjunto (evita vazamento de assunto entre treino e teste).
4. **Representação TF-IDF** ajustada somente com os dados de treino (dentro de um `Pipeline`).
5. **Modelos:** Naive Bayes, Regressão Logística e Random Forest, com hiperparâmetros ajustados por validação cruzada de 5 folds por pares (`GroupKFold` + `GridSearchCV`) e critério F1-score macro.
6. **Controle do comprimento:** classificador de referência que usa apenas o número de palavras, para medir quanto do desempenho vem só do tamanho dos textos.
7. **Protótipo** em Streamlit com o modelo escolhido.

## Resultados

Conjunto final: 3.597 pares (7.194 notícias); treino com 2.877 pares e teste com 720 pares.

**Desempenho no conjunto de teste – versão normalizada (principal)**

| Modelo | Acurácia | F1 macro | Recall (fake) | Taxa de falsos positivos |
|---|---|---|---|---|
| Naive Bayes | 0,889 | 0,889 | 0,903 | 0,125 |
| **Regressão Logística** | **0,929** | **0,929** | **0,935** | **0,078** |
| Random Forest | 0,885 | 0,885 | 0,896 | 0,126 |

**Efeito do comprimento (F1 macro no teste)**

| Modelo | Normalizada | Completa |
|---|---|---|
| Só número de palavras (referência) | 0,503 | 0,938 |
| Regressão Logística | 0,929 | 0,976 |

Na versão completa, o número de palavras sozinho já separa as classes (F1 de 0,938). Por isso, a versão normalizada é a estimativa mais confiável do desempenho. O modelo escolhido para o protótipo foi a **Regressão Logística sem radicalização**.

## Estrutura do repositório

```
Identificador-de-Fake-News/
├── Codigos/
│   ├── analise_exploratoria.py   # análise exploratória (1º bimestre)
│   ├── 02_modelagem.ipynb        # preparação, treinamento e avaliação
│   ├── preprocessamento.py       # carregamento do corpus e limpeza dos textos
│   ├── prototipo.py              # protótipo em Streamlit
│   ├── graficos/                 # gráficos gerados pelos códigos
│   ├── resultados/               # tabelas (CSV) com os resultados
│   └── modelos/                  # modelo treinado (modelo_final.joblib) e metadados
├── Documentacao/                 # artigo e demais documentos
├── requirements.txt
└── README.md
```

## Como executar

**Pré-requisitos:** Python 3.10 ou superior e Git.

### 1. Clonar o repositório e instalar as dependências

```bash
git clone https://github.com/CaioC74/Identificador-de-Fake-News.git
cd Identificador-de-Fake-News
pip install -r requirements.txt
```

### 2. Baixar o corpus

Baixe o Fake.Br Corpus e copie as pastas `full_texts/` e `size_normalized_texts/` para **dentro de `Codigos/`**:

```bash
git clone https://github.com/roneysco/Fake.br-Corpus.git
```

```
Codigos/
├── full_texts/
├── size_normalized_texts/
└── ...
```

Essas pastas estão no `.gitignore` e não são enviadas ao repositório.

### 3. Análise exploratória (opcional)

```bash
cd Codigos
python analise_exploratoria.py
```

Os gráficos são salvos em `Codigos/graficos/` (01 a 07).

### 4. Treinamento e avaliação

Abra `Codigos/02_modelagem.ipynb` no Jupyter ou no VS Code e execute todas as células (**Run All**).

- Para um teste rápido, altere `RAPIDO = True` na seção 1 (grade de hiperparâmetros reduzida, alguns minutos).
- Com `RAPIDO = False`, a execução completa leva entre 10 e 30 minutos, dependendo do computador.
- Na primeira execução, o NLTK baixa automaticamente a lista de stopwords e o stemmer RSLP (requer internet).

Saídas: gráficos 08 a 10 em `graficos/`, tabelas em `resultados/` e o modelo final em `modelos/`.

### 5. Protótipo

O modelo treinado já está em `Codigos/modelos/`, então o protótipo pode ser executado sem rodar o notebook:

```bash
cd Codigos
streamlit run prototipo.py
```

O navegador abre em `http://localhost:8501`. Informe o título e o corpo da notícia e clique em **Analisar**. A tela mostra a probabilidade estimada, a classificação (potencialmente falsa, potencialmente verdadeira ou inconclusiva), as palavras que mais influenciaram a estimativa e as limitações da ferramenta. O texto digitado não é armazenado.

> O modelo foi salvo com o scikit-learn 1.9.1. Use a mesma versão (já fixada no `requirements.txt`) para evitar erros ao carregá-lo.

## Uso de inteligência artificial

Este projeto foi desenvolvido com o auxílio de ferramentas de inteligência artificial, usadas para apoiar a escrita de código.

Todo o conteúdo gerado com ajuda da IA foi revisado, testado e aprovado pela equipe, que é responsável pelas decisões do projeto e pelo resultado final.

## Referências

- Monteiro, R. A., Santos, R. L. S., Pardo, T. A. S., de Almeida, T. A., Ruiz, E. E. S., Vale, O. A. (2018). Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. In: *Computational Processing of the Portuguese Language (PROPOR 2018)*. Lecture Notes in Computer Science, vol. 11122. Springer, Cham.
- Khanam, Z., Alwasel, B. N., El-Zaart, A., Jasser, M. (2021). Fake News Detection Using Machine Learning Approaches. *IOP Conference Series: Materials Science and Engineering*, 1099(1), 012040.
- Sharma, S., Saran, S., Patil, S. (2021). Fake News Detection using Machine Learning Algorithms. *International Journal of Engineering Research & Technology*, 9(3).
- Ahmad, I., Yousaf, M., Yousaf, S., Ahmad, M. O. (2020). Fake News Detection Using Machine Learning Ensemble Methods. *Complexity*, 2020, 8885861.
