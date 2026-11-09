# Estado do projeto

> Projeto de fim de semana, feito por diversão e aprendizado: testar vários modelos de recomendação, brincar
> com um motor adaptativo e aprender técnicas novas. Não é (nem pretende ser) um sistema de produção.

## O que existe
| Parte | Situação |
|---|---|
| Dados sintéticos | 300 lojas (configurável), 200 produtos, 59.023 itens de pedido, 99.235 interações; checagens de qualidade OK |
| Banco | SQLite com 19 tabelas + visão, chaves/restrições/índices, repositórios CRUD |
| Especialistas | content-based, collaborative (kNN e ALS), supervisionado calibrado, bayesiano em dois estágios, popularidade, clusterização |
| Motor adaptativo | regras + gating residual treinável (seleção de complexidade por validação cruzada), explicações |
| Aprendizado | feedback → atualização bayesiana incremental → features → recomendações; retreino versionado |
| Avaliação | protocolo temporal, IC por bootstrap, comparações pareadas, segmentos e cenários |
| Dados reais | mesmo pipeline sobre o Online Retail II (UCI) via perfil de domínio `online_retail` |
| App | Streamlit com 5 páginas |
| Testes | suíte pytest (unidade, integração e ponta a ponta) + GitHub Actions |

## Principais resultados
- **Sintético (273 lojas avaliadas, NDCG@10):** pesos fixos ajustados 0,316 · adaptativo treinado 0,313 ·
  regras 0,302 · supervisionado 0,289. Híbridos > especialistas isolados (IC acima de zero);
  adaptativo × pesos fixos = empate.
- **Efeito do tamanho:** com 50 lojas o gating treinável não aprendia e surgiu uma "vantagem" que era ruído;
  com 300 lojas o gating passou a aprender e a vantagem desapareceu.
- **Dados reais (Online Retail II):** todos os métodos superam o aleatório; as melhores estratégias empatam
  (~0,084–0,094); o bayesiano não ajuda sem logs de interação.

## Ideias para continuar brincando
1. Otimizar o gating diretamente para NDCG (ex.: LambdaRank) ou validar em várias janelas (rolling origin).
2. Melhorar as regras de cold start (o adaptativo perde para pesos fixos em lojas sem histórico).
3. Decaimento temporal da evidência no modelo bayesiano.
4. Um especialista de embeddings (two-tower) usando a interface `BaseRecommender`.
5. Gerar logs de exposição sintéticos para o dataset real e testar o bayesiano lá.

## Notas práticas
- As interações registradas no Simulador ficam gravadas no banco local; para voltar ao estado inicial:
  `python scripts/initialize_database.py --reset` e depois `python scripts/train_models.py`.
- Tamanho da base: `--customers N` em `scripts/generate_data.py` ou variável `ADAPTIVE_RECOMMENDER_N_CUSTOMERS`.
