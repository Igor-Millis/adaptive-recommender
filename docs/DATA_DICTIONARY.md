# Dicionário de Dados

Legenda da origem: **C** = cadastro (declarado/observável no cadastro) · **O** = observado (calculado do
histórico) · **I** = inferido (depende de modelo/hipótese) · **S** = sistema (auditoria/resultado de modelo).

## customers (C)
| Coluna | Tipo | Descrição |
|---|---|---|
| customer_id | INTEGER PK | identificador |
| name | TEXT único | nome fictício da loja |
| city, state | TEXT | cidade e UF |
| registration_date | TIMESTAMP | data de cadastro |
| store_size | Pequeno/Médio/Grande | porte da loja |
| business_segment | TEXT | segmento comercial declarado |
| revenue_band | TEXT | faixa de faturamento |
| declared_categories | TEXT (`;`) | categorias de maior interesse declaradas no cadastro |
| declared_price_band | Econômica/Intermediária/Premium | faixa de preço preferida declarada |
| declared_purchase_frequency | TEXT | frequência de compra declarada |

## products (C)
| Coluna | Descrição |
|---|---|
| product_id, name | identificador e nome |
| category, subcategory | 9 categorias, 42 subcategorias |
| target_audience | Masculino/Feminino/Unissex/Infantil |
| price_band | tercil de preço dentro da categoria |
| price, cost, margin | preço de atacado (R$), custo, margem = (preço − custo)/preço |
| color, size_range, material | atributos físicos |
| season, collection | Verão/Inverno/Atemporal e coleção |
| launch_date | data de lançamento (disponibilidade temporal) |
| stock | estoque atual (≥ 0) |
| description | texto usado pelo TF-IDF |
| is_active | 0/1 |

## orders / order_items (O)
| Coluna | Descrição |
|---|---|
| order_id, customer_id, order_date, channel | cabeçalho do pedido; `source` = synthetic/app |
| item_id, order_id, product_id | linha do pedido |
| quantity (> 0), unit_price (> 0), discount ∈ [0,1) | quantidades e preços |
| total_value | quantity × unit_price × (1 − discount) |
| *category* | disponível na visão `v_order_items_flat` (normalização: categoria pertence ao produto) |

## interactions (O)
| Coluna | Descrição |
|---|---|
| interaction_id, session_id | evento e sessão de recomendação |
| customer_id, product_id | quem e o quê |
| model_name | modelo que recomendou (legado_popularidade, legado_exploracao, hybrid_learned…) |
| interaction_type | impression, click, interest, add_to_cart, purchase, rejection |
| timestamp | data e hora |
| context | JSON (canal, posição, origem, estratégia) |
| recommendation_id | FK opcional para a recomendação exibida pelo sistema |

**Impressão sem resposta não é rejeição**: rejeição só existe como evento explícito.

## customer_features (O + I)
| Feature | Origem | Definição |
|---|---|---|
| n_orders, n_lines, n_distinct_products | O | contagens até o corte |
| total_revenue, avg_ticket | O | faturamento e valor médio por pedido |
| avg_units_per_order, avg_lines_per_order | O | tamanho médio do pedido |
| avg_unit_price, avg_discount | O | médias ponderadas por quantidade |
| recency_days | O | dias desde o último pedido |
| frequency_per_month | O | pedidos / meses ativos |
| tenure_days | O | dias desde o cadastro |
| share_⟨categoria⟩ | O | participação da categoria nas unidades |
| n_impressions, n_engagements, n_rejections, n_rec_purchases | O | desfechos resolvidos de recomendações |
| price_index | I | média ponderada de log(preço pago / mediana da categoria); < 0 ⇒ sensível a preço |
| category_entropy | I | entropia normalizada das participações (diversidade) |
| seasonality_index, seasonal_peak_month | I | 1ª harmônica circular com correção de viés |
| drift_js | I | Jensen-Shannon entre últimos 120 dias e período anterior |
| consistency | I | cosseno entre 1ª e 2ª metade do histórico |
| engagement_rate | I | (engajamentos + 1)/(impressões + 10) |

## Resultados e auditoria (S)
| Tabela | Conteúdo |
|---|---|
| models / model_versions | registro de modelos, versões, corte de treino, parâmetros, métricas, artefato, status ativo/arquivado |
| cluster_runs / cluster_profiles / customer_clusters | execução de clusterização (k, silhouette, diagnósticos), perfis interpretáveis e atribuições |
| bayesian_estimates | α, β (prior e posterior) por cliente × categoria × estágio (interest/conversion) |
| bayesian_update_log | cada atualização incremental: antes/depois, IC, evidência aplicada, motivo |
| recommendation_runs / recommendations | o que foi recomendado, com score, probabilidade, contribuições e motivos |
| expert_weights | pesos efetivamente usados em cada execução + sinais |
| evaluation_runs / evaluation_results | protocolo e métricas (com IC) por estágio, modelo, segmento e K |
| event_log | eventos do sistema (carga, feedback, retreino, avaliação) |

## Arquivos
| Arquivo | Conteúdo |
|---|---|
| data/synthetic/*.csv | dados brutos gerados |
| data/synthetic/ground_truth_customers.json | arquétipos e parâmetros latentes (somente análise) |
| data/synthetic/ground_truth_products.csv | estilo e qualidade latente dos produtos (somente análise) |
| models/trained/supervised_vN.joblib (+ _report.json) | modelo, calibrador e relatório de treino |
| models/trained/gating_vN.json | parâmetros do gating treinável e pesos fixos |
| models/metadata/evaluation_latest.json | resultado completo do último experimento |
