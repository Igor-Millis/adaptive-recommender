# Metodologia

## 1. Problema

Recomendar, para cada lojista B2B, produtos que ele **ainda não comprou** e que tem maior probabilidade de
comprar nos próximos 92 dias, explicando a recomendação e aprendendo com a resposta do cliente.

## 2. Dados sintéticos com estrutura controlada

O gerador (`src/data_generation/`) cria uma *utilidade latente* por cliente e produto:

$$U(c,p,t) = 1{,}3\log\pi^{cat}_{c}(t) + \log\pi^{estilo}_c(t) + 0{,}6\log\pi^{público}_c - 1{,}6\,\lambda_c\log\frac{\text{preço}_p}{\text{mediana}} + 0{,}5\,\mathbb 1[\text{faixa}] + 0{,}8\,\text{estação}_p\cos\tfrac{2\pi(m-1)}{12} + q_p + 0{,}35\,\text{novidade}_p + 0{,}9\,\text{recompra}$$

- **Arquétipos** (ground truth): esportivo, econômico (sensível a preço), social, sazonal (praia),
  diversificado, premium (alto ticket), migrante (deriva gradual casual → esportivo) e complementar
  (compra pares: calça+camisa, vestido+acessório...).
- Preferências por categoria/estilo ~ Dirichlet concentrada no arquétipo (perfis coerentes, não idênticos).
- Pedidos: Poisson mensal com modulação sazonal $1 + A\cos(2\pi(m - m^\star)/12)$ e rampa para novos clientes;
  itens escolhidos por Gumbel-top-k sobre $U$ (equivale a amostragem sem reposição de um logit
  multinomial) + complementares.
- **Tamanho e proporções**: `N_CUSTOMERS` clientes (padrão 300) distribuídos entre os arquétipos na proporção
  da base de referência. **Cold start** proporcional: 4% sem nenhum pedido, 6% cadastrados durante a janela de
  teste, 4% durante a validação, 6% com pouco histórico e 6% com histórico moderado (mínimo de 2 em cada grupo).
  Com 300 clientes: 12 sem pedidos, 18 sem histórico no corte do teste.
- **Interações**: em cada pedido (e em sessões de navegação) um recomendador legado exibe 6 produtos
  (70% populares, 30% exploração). A resposta segue o funil
  $P(\text{engajar}) = \sigma(-2 + 1{,}5z)$, $P(\text{comprar}\mid\text{engajar}) = \sigma(-0{,}6 + z)$,
  $P(\text{rejeitar}\mid\neg\text{engajar}) = \sigma(-3{,}2-1{,}2z)$, com $z$ = utilidade padronizada.
  Compras via recomendação viram itens do próprio pedido (coerência entre tabelas).
- O ground truth fica em `data/synthetic/ground_truth_*` e **nunca** é lido por modelos; é usado apenas para
  validar o gerador (testes) e como segmentação opcional de análise na avaliação.

Validação da coerência: `data_quality_report` (chaves, totais, compras após lançamento e cadastro,
compras-por-recomendação presentes nos pedidos) e testes de padrão identificável (esportivos compram > 2× mais
roupas esportivas que os demais).

## 3. Engenharia de atributos

Sempre ponto-no-tempo (`DataSnapshot`). Separação explícita: **cadastro** (declarado), **observado**
(agregados do histórico) e **inferido** (índices dependentes de hipóteses) — refletida na interface por
selos coloridos. Ver `DATA_DICTIONARY.md`.

## 4. Especialistas

| Especialista | Pergunta que responde | Quando é forte | Quando falha |
|---|---|---|---|
| Popularidade | o que a base está comprando? | cold start | ignora o perfil |
| Content-Based | o que se parece com o que ele compra? | histórico consistente, produtos novos | não descobre categorias novas |
| Collaborative (kNN) | o que compram lojistas parecidos? | vizinhos fortes | cold start, itens novos |
| Supervisionado | qual a probabilidade de compra? | combina todos os sinais; generaliza | depende de rótulos suficientes |
| Bayesiano | como ele responde a recomendações da categoria? | muitas interações | não diferencia produtos da categoria |
| ALS (comparação) | fatores latentes | matrizes grandes e esparsas | aqui perde para kNN (ver avaliação) |

## 5. Motor adaptativo

Sinais de evidência → regras interpretáveis → gating residual treinável (com seleção de complexidade por CV)
→ combinação convexa dos scores normalizados → ranking com regras de negócio → explicação
(pesos, justificativa, especialistas descartados, confiança agregada, sinais).

## 6. Aprendizado contínuo

| Evento | Atualização imediata | Rotina de retreino |
|---|---|---|
| Impressão/clique/interesse/carrinho/rejeição | Δ Beta (conjugado), features do cliente, perfil CB, linha CF, sinais/pesos | priors EB, supervisionado, ALS, clusters, gating |
| Compra | + pedido e baixa de estoque | idem |

Retreino: lock de execução única, novas versões em `model_versions`, versões anteriores arquivadas,
dados operacionais intocados (testado).

## 7. Avaliação experimental

Protocolo temporal treino/validação/teste (ver `MODEL_EVALUATION.md`), mesmos candidatos e janelas para
todos os modelos, IC por bootstrap de clientes, comparações pareadas, análise por disponibilidade de
histórico, por cluster, por cenário detectado (sem histórico, mudança de preferência, sazonal, estável) e
por arquétipo sintético; análise de seleção (o especialista de maior peso é o melhor para o cliente?).

## 8. Hipóteses estatísticas assumidas

1. Estacionariedade local: o comportamento na janela de 92 dias é previsível a partir do passado imediato.
2. Clientes independentes entre si (bootstrap por cliente); pares de um mesmo cliente são correlacionados
   (por isso o bootstrap é por cliente, não por par).
3. Evidência fracionária como verossimilhança temperada (Beta-Binomial).
4. Independência entre estágios do funil e entre categorias no modelo bayesiano.
5. Subamostragem de negativos com pesos inversos preserva a calibração (Horvitz-Thompson).
6. Rótulos de descoberta: comprar um produto novo na janela = relevância; não comprar ≠ desinteresse
   definitivo (tratado como negativo **da janela**).

## 9. Requisitos reavaliados

- *"Recomendar com base em cliente × produto no Bayes"*: rejeitado por insuficiência de dados; adotado
  cliente × categoria (justificado em `MATHEMATICAL_FOUNDATIONS.md` §10).
- *"Produtos nunca comprados como negativos"*: substituído por negativos por janela, clientes ativos e
  features de exposição.
- *"Pesos fixos e iguais como baseline"*: incluídos os dois — iguais e ajustados na validação —, este último
  sendo um baseline mais exigente e mais honesto.
