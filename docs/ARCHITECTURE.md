# Arquitetura

## Visão geral

```
                ┌──────────────────────────── app/ (Streamlit: só apresentação) ───────────────────────────┐
                │ Dashboard │ Clientes │ Recomendações │ Simulador │ Avaliação      components/ (charts, cards…) │
                └──────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                       │ chama apenas serviços
                ┌──────────────────────────── src/services/ ──────────────────────────────────────────────┐
                │ RecommendationEngine · CustomerService · AnalyticsService · ModelService               │
                └───────┬───────────────────────┬──────────────────────┬──────────────────────┬───────────┘
                        │                       │                      │                      │
     src/adaptive/ (gating, MoE)   src/ranking/ (filtros, MMR,   src/learning/ (feedback,   src/evaluation/
                        │           margem)                     Bayes incremental, retreino) (temporal, métricas)
                        │                                              │
     src/recommenders/ (popularidade, content-based, user-kNN, ALS, supervisionado, bayesiano, suite)
                        │
     src/segmentation/ (K-Means, Ward)      src/preprocessing/ (snapshot ponto-no-tempo, features)
                        │
     database/ (schema.sql, connection, repositories)  ←→  SQLite (data/adaptive_recommender.db)
     src/data_generation/ (gerador sintético → data/synthetic/*.csv)
```

Regras de dependência: `app` → `services` → (`adaptive`, `ranking`, `learning`, `evaluation`) →
`recommenders` → (`segmentation`, `preprocessing`) → `database`. Nenhum módulo de `src/` importa
Streamlit; toda a camada de recomendação roda em scripts e testes sem interface.

## Estrutura de diretórios

| Pasta | Conteúdo |
|---|---|
| `app/` | `main.py` (navegação), `views/` (5 páginas), `components/` (tema, cards, gráficos, tabelas, filtros, estado) |
| `config/settings.py` | caminhos, datas do protocolo temporal, hiperparâmetros (dataclasses imutáveis) |
| `database/` | `schema.sql`, `connection.py`, `init_db.py`, `repositories/` (CRUD parametrizado) |
| `data/synthetic/` | CSVs gerados + *ground truth* latente (somente análise) |
| `data/processed/` | reservado para extrações processadas (features persistem na tabela `customer_features`) |
| `src/data_generation/` | clientes, produtos, simulação cronológica de pedidos e interações |
| `src/preprocessing/` | snapshots temporais, limpeza, engenharia de atributos, transformações |
| `src/segmentation/` | clusterização e interpretação dos segmentos |
| `src/recommenders/` | especialistas com interface comum (`base.py`) e `suite.py` |
| `src/adaptive/` | sinais e regras (`model_selection.py`), gating (`gating.py`), combinação/explicação (`ensemble.py`) |
| `src/ranking/` | elegibilidade, fadiga, margem, MMR |
| `src/learning/` | feedback, atualização bayesiana incremental, retreinamento controlado |
| `src/evaluation/` | métricas, protocolo temporal, comparação, experimento completo |
| `src/services/` | fachadas usadas pela interface |
| `models/trained/` | artefatos versionados (`supervised_vN.joblib`, `gating_vN.json`) |
| `models/metadata/` | `evaluation_latest.json` (resultado completo do experimento) |
| `scripts/` | `generate_data`, `initialize_database`, `train_models`, `run_evaluation` |
| `tests/` | pytest (dados, banco, especialistas, Bayes, adaptativo, avaliação, ponta a ponta) |

## Modelo de dados

| Camada | Tabelas |
|---|---|
| Operacional (bruto) | `customers`, `products`, `orders`, `order_items`, `interactions` |
| Processado | `customer_features` |
| Resultados de modelos | `models`, `model_versions`, `cluster_runs`, `cluster_profiles`, `customer_clusters`, `bayesian_estimates`, `recommendation_runs`, `recommendations`, `expert_weights`, `evaluation_runs`, `evaluation_results` |
| Auditoria | `bayesian_update_log`, `event_log` |
| Visão | `v_order_items_flat` (itens com categoria e data) |

Integridade: chaves primárias/estrangeiras com `PRAGMA foreign_keys = ON`, `ON DELETE RESTRICT` para
entidades com histórico, `CHECK` para domínios (tipos de interação, estoque ≥ 0, probabilidades e pesos em
[0, 1], desconto em [0, 1)), índices em chaves estrangeiras e datas. Detalhes em `DATA_DICTIONARY.md`.

O que é persistido × recalculado: scores dos especialistas determinísticos (CB, CF, popularidade, ALS) são
recalculados ao carregar o motor (< 2 s) e **não** são armazenados; são persistidos apenas estados que não
podem ser recalculados sem perda (estado bayesiano incremental), artefatos treinados versionados e registros
de auditoria (recomendações exibidas, pesos usados, eventos).

## Fluxo de processamento

1. **Geração** (`scripts/generate_data.py`) → CSVs + relatório de qualidade.
2. **Carga** (`scripts/initialize_database.py`) → esquema + dados (não sobrescreve banco populado).
3. **Treino** (`scripts/train_models.py` → `learning/retraining.py`): features → clusterização → supervisionado
   (snapshots, holdout, calibração) → priors bayesianos → gating na janela mais recente → versões.
4. **Serviço** (`RecommendationEngine`): carrega versões ativas, ajusta especialistas determinísticos,
   calcula sinais → gating → combinação → filtros → ranking/MMR → explicações → log da execução.
5. **Feedback** (`FeedbackService`): interação → (pedido) → Δ bayesiano → features → refresh incremental.
6. **Avaliação** (`scripts/run_evaluation.py`): protocolo temporal completo → banco + JSON + `MODEL_EVALUATION.md`.

## Perfis de domínio e dados reais

`config/settings.py` lê o perfil em `ADAPTIVE_RECOMMENDER_PROFILE` (padrão `synthetic`). Um perfil
(`config/profiles/<nome>.json`) troca categorias, datas do protocolo temporal, horizonte, hemisfério
(sazonalidade de calendário), hiperparâmetros do supervisionado e os arquivos de saída da avaliação.
O perfil `online_retail` executa o MESMO pipeline sobre o dataset público Online Retail II:

- `scripts/download_real_data.py` baixa o zip da UCI e converte para CSV em `data/raw/online_retail_ii/`;
- `src/data_sources/online_retail.py` limpa (cancelamentos, devoluções, clientes anônimos, códigos
  administrativos), deriva categorias/atributos da descrição, seleciona o catálogo só com dados anteriores
  à validação e amostra clientes;
- `scripts/run_real_data_experiment.py` roda o experimento em memória e grava
  `docs/REAL_DATA_EVALUATION.md` e `models/metadata/evaluation_online_retail.json`, sem tocar no banco nem
  nos resultados sintéticos.

## Extensibilidade (Deep Learning)

Um novo especialista (ex.: embeddings two-tower, modelo sequencial) precisa apenas implementar
`BaseRecommender` (`fit`, `score`, `confidence`, `explain`) e ser adicionado a `ExpertSuite` e à lista
`EXPERTS`. O gating é dimensionado pelo número de especialistas (softmax sobre $E$ logits) e a avaliação,
a interface de pesos e as explicações iteram sobre `EXPERTS` — nada mais muda.

## Migração para PostgreSQL

- `DATABASE_URL` pode ser definida por variável de ambiente (`ADAPTIVE_RECOMMENDER_DB_URL`).
- O SQL usa `INSERT ... RETURNING` e `ON CONFLICT ... DO UPDATE` (suportados por ambos).
- Ajustes necessários: `INTEGER PRIMARY KEY` → `GENERATED ALWAYS AS IDENTITY`; `MAX(a, b)` escalar
  (usado na baixa de estoque) → `GREATEST`; colunas JSON em `TEXT` → `JSONB`; remover os `PRAGMA`
  (já condicionados a SQLite em `connection.py`); `table_exists` já usa o *inspector* do SQLAlchemy.

## Decisões técnicas e consequências

| Decisão | Motivo | Consequência |
|---|---|---|
| Popularidade como 5º especialista | fallback explícito no cold start | soma dos pesos continua 1; documentado como extensão da especificação |
| Bayes em cliente × categoria, 2 estágios | dados insuficientes por produto; separa interesse de conversão | não discrimina produtos dentro da categoria (complementado pelos demais) |
| Tarefa de avaliação = descoberta | recompra é trivial e domina o histórico (~72% das linhas do teste) | métricas medem a capacidade de sugerir algo novo |
| Normalização por percentil | escalas incomparáveis entre especialistas | perde magnitude; confiança tratada no gating |
| Gating residual sobre regras + seleção por CV | poucos clientes para aprender 60 parâmetros | com 50 clientes recaía nas regras; com 300 aprende ajustes (variante viés+contexto) |
| Estoque fora da avaliação | não há histórico de estoque | avaliação não penaliza itens hoje esgotados |
| Implementação própria de ALS | evita dependência compilada extra | adequado ao porte; não escala para milhões de linhas |
