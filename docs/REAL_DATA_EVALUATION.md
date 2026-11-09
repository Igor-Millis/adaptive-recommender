# Avaliação em dados reais — Online Retail II (UCI)

> Documento **gerado automaticamente** por `scripts/run_evaluation.py` a partir dos resultados reais do experimento.
> Reexecutar o script atualiza todos os números.

## 1. Protocolo

- Treino: dados < 2011-06-13; validação [2011-06-13, 2011-09-11); teste [2011-09-11, 2011-12-10).
- Tarefa: **descoberta (produtos não comprados até o corte)**; relevante = produto comprado na janela de 90 dias.
- Especialistas re-ajustados com dados anteriores ao teste; gating treinável e pesos fixos aprendidos **somente** na validação.
- Clientes avaliados no teste: 218 (com ≥ 1 compra nova na janela). IC95% por bootstrap sobre clientes (2000 reamostragens).
- Pesos fixos ajustados na validação (grade no simplex, passo 0,1): Content-Based 0.0, Collaborative Filtering 0.0, Supervisionado 0.9, Bayesiano 0.0, Popularidade 0.1.

## 2. Ranking — teste (K = 10)

| Estratégia | ndcg | precision | recall | map | hit_rate | coverage | diversity |
|---|---|---|---|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.0938 [0.076, 0.112] | 0.0656 [0.053, 0.078] | 0.0864 [0.066, 0.108] | 0.0443 [0.033, 0.056] | 0.4174 [0.353, 0.482] | 0.3900 | 0.5279 [0.513, 0.542] |
| Supervisionado | 0.0912 [0.072, 0.111] | 0.0606 [0.049, 0.073] | 0.0824 [0.062, 0.105] | 0.0446 [0.032, 0.058] | 0.3899 [0.326, 0.454] | 0.6575 | 0.5204 [0.505, 0.535] |
| Collaborative Filtering | 0.0851 [0.069, 0.101] | 0.0633 [0.052, 0.076] | 0.0899 [0.069, 0.112] | 0.0370 [0.028, 0.047] | 0.4312 [0.367, 0.495] | 0.8450 | 0.5835 [0.573, 0.593] |
| Híbrido pesos iguais | 0.0851 [0.068, 0.104] | 0.0674 [0.055, 0.081] | 0.0871 [0.066, 0.110] | 0.0377 [0.028, 0.049] | 0.4174 [0.353, 0.482] | 0.3000 | 0.4928 [0.481, 0.503] |
| Adaptativo (gating treinado) | 0.0835 [0.066, 0.102] | 0.0647 [0.053, 0.078] | 0.0876 [0.066, 0.111] | 0.0366 [0.027, 0.047] | 0.4128 [0.349, 0.477] | 0.3800 | 0.4785 [0.467, 0.489] |
| Fatoração de Matrizes (ALS) | 0.0807 [0.063, 0.100] | 0.0592 [0.047, 0.072] | 0.0802 [0.061, 0.102] | 0.0384 [0.026, 0.053] | 0.4083 [0.339, 0.473] | 0.9475 | 0.5967 [0.589, 0.604] |
| Adaptativo (regras) | 0.0803 [0.062, 0.099] | 0.0610 [0.049, 0.074] | 0.0793 [0.058, 0.102] | 0.0367 [0.026, 0.048] | 0.3899 [0.326, 0.454] | 0.6675 | 0.4306 [0.417, 0.444] |
| Bayesiano | 0.0735 [0.058, 0.089] | 0.0583 [0.046, 0.072] | 0.0581 [0.044, 0.072] | 0.0322 [0.024, 0.041] | 0.3670 [0.303, 0.431] | 0.0875 | 0.5216 [0.514, 0.529] |
| Popularidade | 0.0698 [0.055, 0.085] | 0.0601 [0.047, 0.074] | 0.0581 [0.045, 0.073] | 0.0294 [0.022, 0.038] | 0.3670 [0.303, 0.427] | 0.0850 | 0.4908 [0.483, 0.499] |
| Content-Based | 0.0690 [0.053, 0.086] | 0.0523 [0.041, 0.064] | 0.0679 [0.050, 0.088] | 0.0311 [0.022, 0.042] | 0.3578 [0.294, 0.422] | 0.8625 | 0.3861 [0.368, 0.404] |
| Aleatório | 0.0337 [0.024, 0.044] | 0.0280 [0.021, 0.036] | 0.0260 [0.017, 0.035] | 0.0125 [0.008, 0.017] | 0.2202 [0.165, 0.280] | 1.0000 | 0.6564 [0.652, 0.661] |

### K = 5
| Estratégia | ndcg | precision | recall |
|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.0867 [0.067, 0.108] | 0.0679 [0.053, 0.083] | 0.0517 [0.036, 0.069] |
| Supervisionado | 0.0844 [0.063, 0.107] | 0.0651 [0.050, 0.082] | 0.0414 [0.028, 0.055] |
| Adaptativo (regras) | 0.0729 [0.054, 0.094] | 0.0670 [0.050, 0.085] | 0.0454 [0.029, 0.064] |
| Bayesiano | 0.0729 [0.055, 0.092] | 0.0633 [0.048, 0.080] | 0.0348 [0.024, 0.047] |
| Collaborative Filtering | 0.0728 [0.056, 0.090] | 0.0661 [0.051, 0.081] | 0.0446 [0.031, 0.060] |
| Fatoração de Matrizes (ALS) | 0.0726 [0.053, 0.094] | 0.0633 [0.049, 0.079] | 0.0458 [0.030, 0.065] |
| Híbrido pesos iguais | 0.0716 [0.053, 0.092] | 0.0661 [0.050, 0.084] | 0.0447 [0.030, 0.063] |
| Adaptativo (gating treinado) | 0.0713 [0.054, 0.091] | 0.0651 [0.050, 0.083] | 0.0434 [0.029, 0.061] |
| Popularidade | 0.0636 [0.047, 0.080] | 0.0615 [0.046, 0.078] | 0.0304 [0.021, 0.041] |
| Content-Based | 0.0612 [0.045, 0.078] | 0.0569 [0.043, 0.072] | 0.0394 [0.027, 0.053] |
| Aleatório | 0.0338 [0.022, 0.048] | 0.0321 [0.021, 0.044] | 0.0149 [0.009, 0.021] |

### K = 20
| Estratégia | ndcg | precision | recall |
|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.1160 [0.098, 0.135] | 0.0594 [0.050, 0.069] | 0.1600 [0.133, 0.189] |
| Supervisionado | 0.1143 [0.095, 0.134] | 0.0576 [0.048, 0.068] | 0.1554 [0.128, 0.183] |
| Híbrido pesos iguais | 0.1050 [0.088, 0.123] | 0.0596 [0.050, 0.070] | 0.1513 [0.124, 0.180] |
| Fatoração de Matrizes (ALS) | 0.1040 [0.086, 0.123] | 0.0550 [0.045, 0.065] | 0.1547 [0.127, 0.183] |
| Adaptativo (gating treinado) | 0.1038 [0.086, 0.122] | 0.0583 [0.049, 0.068] | 0.1509 [0.123, 0.178] |
| Collaborative Filtering | 0.1029 [0.087, 0.119] | 0.0525 [0.044, 0.062] | 0.1525 [0.125, 0.182] |
| Adaptativo (regras) | 0.0973 [0.080, 0.116] | 0.0525 [0.044, 0.061] | 0.1378 [0.110, 0.167] |
| Bayesiano | 0.0922 [0.077, 0.108] | 0.0550 [0.046, 0.065] | 0.1217 [0.099, 0.147] |
| Popularidade | 0.0853 [0.071, 0.101] | 0.0534 [0.044, 0.064] | 0.1138 [0.091, 0.138] |
| Content-Based | 0.0836 [0.067, 0.101] | 0.0447 [0.036, 0.054] | 0.1165 [0.091, 0.143] |
| Aleatório | 0.0401 [0.031, 0.049] | 0.0255 [0.020, 0.031] | 0.0494 [0.038, 0.062] |

### Tempo de execução (segundos, ajuste + scoring no estágio de teste)
- Popularidade: 0.01s
- Content-Based: 0.63s
- Collaborative Filtering: 0.03s
- Fatoração de Matrizes (ALS): 0.68s
- Supervisionado: 0.92s
- Bayesiano: 0.05s
- Híbrido pesos iguais: 12.11s
- Híbrido pesos fixos (ajustados): 12.11s
- Adaptativo (regras): 12.11s
- Adaptativo (gating treinado): 12.11s

## 3. Comparações pareadas (ΔNDCG@10, bootstrap pareado)

| A | B | Δ (A−B) | IC95% | P(Δ ≤ 0) |
|---|---|---|---|---|
| Adaptativo (gating treinado) | Híbrido pesos fixos (ajustados) | -0.0103 | [-0.0253, +0.0045] | 0.907 |
| Adaptativo (regras) | Híbrido pesos fixos (ajustados) | -0.0136 | [-0.0287, +0.0020] | 0.958 |
| Adaptativo (gating treinado) | Adaptativo (regras) | +0.0033 | [-0.0046, +0.0105] | 0.192 |
| Adaptativo (gating treinado) | Supervisionado | -0.0077 | [-0.0242, +0.0089] | 0.805 |
| Adaptativo (regras) | Supervisionado | -0.0110 | [-0.0276, +0.0051] | 0.903 |
| Híbrido pesos fixos (ajustados) | Supervisionado | +0.0026 | [-0.0059, +0.0103] | 0.267 |
| Adaptativo (gating treinado) | Híbrido pesos iguais | -0.0015 | [-0.0058, +0.0031] | 0.764 |
| Supervisionado | Popularidade | +0.0214 | [+0.0003, +0.0428] | 0.025 |
| Collaborative Filtering | Fatoração de Matrizes (ALS) | +0.0044 | [-0.0189, +0.0249] | 0.349 |

## 4. Por disponibilidade de histórico (NDCG@10)

| Estratégia | histórico extenso (n=25) | histórico moderado (n=56) | pouco histórico (n=79) | sem histórico (n=58) |
|---|---|---|---|---|
| Híbrido pesos iguais | 0.0903 | 0.0857 | 0.0959 | 0.0673 |
| Collaborative Filtering | 0.0878 | 0.0954 | 0.0792 | 0.0820 |
| Adaptativo (gating treinado) | 0.0872 | 0.0859 | 0.0923 | 0.0676 |
| Adaptativo (regras) | 0.0703 | 0.0908 | 0.0868 | 0.0655 |
| Fatoração de Matrizes (ALS) | 0.0570 | 0.0813 | 0.0868 | 0.0820 |
| Bayesiano | 0.0558 | 0.0860 | 0.0569 | 0.0918 |
| Híbrido pesos fixos (ajustados) | 0.0513 | 0.1197 | 0.1001 | 0.0786 |
| Content-Based | 0.0454 | 0.0732 | 0.0638 | 0.0820 |
| Popularidade | 0.0448 | 0.0853 | 0.0610 | 0.0776 |
| Supervisionado | 0.0349 | 0.1191 | 0.0998 | 0.0769 |
| Aleatório | 0.0228 | 0.0311 | 0.0327 | 0.0422 |

### Por cenário detectado nos dados observados (NDCG@10)

| Estratégia | mudança de preferência (n=29) | padrão estável (n=113) | sazonal (n=18) | sem histórico (n=58) |
|---|---|---|---|---|
| Supervisionado | 0.1400 | 0.0949 | 0.0357 | 0.0769 |
| Híbrido pesos fixos (ajustados) | 0.1275 | 0.0992 | 0.0551 | 0.0786 |
| Adaptativo (regras) | 0.1143 | 0.0814 | 0.0657 | 0.0655 |
| Adaptativo (gating treinado) | 0.1058 | 0.0893 | 0.0629 | 0.0676 |
| Collaborative Filtering | 0.0954 | 0.0844 | 0.0833 | 0.0820 |
| Híbrido pesos iguais | 0.0939 | 0.0939 | 0.0725 | 0.0673 |
| Fatoração de Matrizes (ALS) | 0.0829 | 0.0872 | 0.0321 | 0.0820 |
| Bayesiano | 0.0793 | 0.0589 | 0.0971 | 0.0918 |
| Content-Based | 0.0712 | 0.0578 | 0.0934 | 0.0820 |
| Popularidade | 0.0707 | 0.0629 | 0.0867 | 0.0776 |
| Aleatório | 0.0514 | 0.0249 | 0.0331 | 0.0422 |

Cenários: *sem histórico* (0 pedidos no corte), *mudança de preferência* (JS > 0,15), *sazonal* (índice > 0,06), *padrão estável* (demais). Segmentos com n < 10 são indicativos, não conclusivos.

## 5. Seleção de especialistas pelo motor adaptativo

Pesos médios no teste:

| Gating | Content-Based | Collaborative Filtering | Supervisionado | Bayesiano | Popularidade |
|---|---|---|---|---|---|
| Híbrido pesos iguais | 0.18 | 0.16 | 0.22 | 0.22 | 0.22 |
| Híbrido pesos fixos (ajustados) | 0.00 | 0.00 | 0.90 | 0.00 | 0.10 |
| Adaptativo (regras) | 0.36 | 0.14 | 0.30 | 0.03 | 0.17 |
| Adaptativo (gating treinado) | 0.22 | 0.14 | 0.23 | 0.27 | 0.14 |

Gating treinável — seleção de complexidade por validação cruzada (5 dobras sobre clientes da validação;
empates de ±0,001 favorecem a variante mais simples):

| Variante | λ (L2 de θ) | NDCG@10 fora da dobra |
|---|---|---|
| regras | — | 0.1090 ± 0.0483 |
| viés | — | 0.1146 ± 0.0503 |
| viés+contexto | 0.03 | 0.1067 ± 0.0448 |
| viés+contexto | 0.1 | 0.1107 ± 0.0495 |
| viés+contexto | 0.3 | 0.1092 ± 0.0486 |
| viés+contexto | 1.0 | 0.1130 ± 0.0519 |
| viés+contexto | 3.0 | 0.1149 ± 0.0508 |
| viés+contexto | 10.0 | 0.1156 ± 0.0497 |

Variante escolhida: **viés** (λ = None).

## 6. Modelo supervisionado — classificação no teste

| Algoritmo | ROC-AUC | PR-AUC | Log Loss | Brier | Precisão | Recall | Taxa base |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.6976 | 0.0472 | 0.1025 | 0.0216 | 0.055 | 0.253 | 0.0223 |
| random_forest | 0.6527 | 0.0421 | 0.1037 | 0.0217 | 0.058 | 0.113 | 0.0223 |
| lightgbm (selecionado) | 0.6874 | 0.0457 | 0.1026 | 0.0217 | 0.060 | 0.120 | 0.0223 |

Limiar de decisão definido no *holdout* do treino (máximo F1) e reaplicado no teste. Probabilidades calibradas por Platt.

## 7. Clusterização (corte do teste)

| k | Inertia | Silhouette | Estabilidade (ARI bootstrap) |
|---|---|---|---|
| 2 | 1943.1 | 0.202 | 0.684 |
| 3 | 1729.9 | 0.127 | 0.521 |
| 4 | 1566.6 | 0.147 | 0.444 |
| 5 | 1438.5 | 0.140 | 0.655 |
| 6 | 1359.5 | 0.108 | 0.524 |
| 7 | 1294.1 | 0.128 | 0.550 |
| 8 | 1250.3 | 0.098 | 0.474 |
| 9 | 1205.9 | 0.111 | 0.507 |
| 10 | 1172.2 | 0.107 | 0.474 |

k escolhido = 5. Silhouette Ward = 0.123; ARI K-Means × Ward = 0.608.

## 8. Conclusões (geradas a partir dos números acima)

- Melhor NDCG@10 no teste: Híbrido pesos fixos (ajustados) = 0.0938 (IC95% 0.0757–0.1124).
- Adaptativo (gating treinado) − Híbrido pesos fixos (ajustados): ΔNDCG@10 = -0.0103 (IC95% -0.0253 a +0.0045) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (regras) − Híbrido pesos fixos (ajustados): ΔNDCG@10 = -0.0136 (IC95% -0.0287 a +0.0020) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (gating treinado) − Adaptativo (regras): ΔNDCG@10 = +0.0033 (IC95% -0.0046 a +0.0105) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (gating treinado) − Supervisionado: ΔNDCG@10 = -0.0077 (IC95% -0.0242 a +0.0089) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (regras) − Supervisionado: ΔNDCG@10 = -0.0110 (IC95% -0.0276 a +0.0051) → diferença NÃO conclusiva (IC95% inclui zero).
- Híbrido pesos fixos (ajustados) − Supervisionado: ΔNDCG@10 = +0.0026 (IC95% -0.0059 a +0.0103) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (gating treinado) − Híbrido pesos iguais: ΔNDCG@10 = -0.0015 (IC95% -0.0058 a +0.0031) → diferença NÃO conclusiva (IC95% inclui zero).
- Supervisionado − Popularidade: ΔNDCG@10 = +0.0214 (IC95% +0.0003 a +0.0428) → ganho estatisticamente consistente (IC95% acima de zero).
- Collaborative Filtering − Fatoração de Matrizes (ALS): ΔNDCG@10 = +0.0044 (IC95% -0.0189 a +0.0249) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (regras): o especialista de maior peso coincide com o melhor especialista do cliente em 47% dos casos (oráculo NDCG@10 = 0.1726).
- Adaptativo (gating treinado): o especialista de maior peso coincide com o melhor especialista do cliente em 46% dos casos (oráculo NDCG@10 = 0.1726).

## 9. Limitações e interpretação

- Os dados são sintéticos: o gerador define a utilidade latente; ganhos aqui não garantem ganhos em dados reais.
- Clientes avaliados: 218 no teste e 138 na validação
  (onde o gating é aprendido). A largura dos IC95% acima indica a menor diferença que este tamanho de amostra
  consegue distinguir do ruído.
- O gating treinável é regularizado em direção às regras e só substitui as regras quando vence na validação
  cruzada; quando isso não ocorre, ele reproduz exatamente o gating por regras.
- O especialista bayesiano opera no nível de categoria (por desenho): é útil como sinal de resposta a
  recomendações, mas não discrimina produtos dentro da categoria.

## 10. Dados reais utilizados

- Fonte: Online Retail II — UCI Machine Learning Repository (CC BY 4.0), dez/2009 a dez/2011.
- Linhas brutas: 1,067,371; após limpeza (sem cancelamentos, devoluções, clientes anônimos e códigos administrativos): 802,632.
- Amostra: 500 clientes aleatórios e os 400 produtos com mais compradores antes da validação → 3,052 pedidos, 33,379 linhas.
- Clientes cuja 1ª compra ocorre no teste (cold start real): 58.
- Produtos por categoria (derivada da descrição): Cozinha e confeitaria 89, Decoração da casa 65, Outros 65, Jardim e organização 45, Bolsas e sacolas 35, Festas e presentes 32, Papelaria e cartões 22, Infantil e brinquedos 19, Velas e iluminação 16, Natal e datas sazonais 12.
- Sem logs de interação: especialista bayesiano opera apenas com o prior.
- Categorias derivadas por palavras-chave da descrição.
- Custo/margem e estoque desconhecidos (não afetam métricas de relevância).
