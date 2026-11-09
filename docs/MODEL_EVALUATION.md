# Avaliação dos Modelos

> Documento **gerado automaticamente** por `scripts/run_evaluation.py` a partir dos resultados reais do experimento.
> Reexecutar o script atualiza todos os números.

## 1. Protocolo

- Treino: dados < 2026-05-01; validação [2026-05-01, 2026-08-01); teste [2026-08-01, 2026-11-01).
- Tarefa: **descoberta (produtos não comprados até o corte)**; relevante = produto comprado na janela de 92 dias.
- Especialistas re-ajustados com dados anteriores ao teste; gating treinável e pesos fixos aprendidos **somente** na validação.
- Clientes avaliados no teste: 273 (com ≥ 1 compra nova na janela). IC95% por bootstrap sobre clientes (2000 reamostragens).
- Pesos fixos ajustados na validação (grade no simplex, passo 0,1): Content-Based 0.1, Collaborative Filtering 0.6, Supervisionado 0.3, Bayesiano 0.0, Popularidade 0.0.

## 2. Ranking — teste (K = 10)

| Estratégia | ndcg | precision | recall | map | hit_rate | coverage | diversity |
|---|---|---|---|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.3160 [0.287, 0.343] | 0.1890 [0.171, 0.207] | 0.3506 [0.320, 0.381] | 0.1923 [0.170, 0.215] | 0.8132 [0.766, 0.857] | 0.8824 | 0.7656 [0.760, 0.771] |
| Adaptativo (gating treinado) | 0.3133 [0.286, 0.340] | 0.1864 [0.168, 0.205] | 0.3557 [0.323, 0.388] | 0.1901 [0.168, 0.213] | 0.8205 [0.777, 0.864] | 0.8556 | 0.7458 [0.739, 0.752] |
| Adaptativo (regras) | 0.3023 [0.276, 0.329] | 0.1828 [0.166, 0.200] | 0.3518 [0.320, 0.382] | 0.1790 [0.158, 0.202] | 0.8425 [0.795, 0.883] | 0.9198 | 0.7431 [0.737, 0.749] |
| Híbrido pesos iguais | 0.2897 [0.264, 0.316] | 0.1762 [0.159, 0.193] | 0.3378 [0.308, 0.369] | 0.1682 [0.148, 0.189] | 0.8352 [0.791, 0.875] | 0.7807 | 0.7588 [0.754, 0.764] |
| Supervisionado | 0.2887 [0.263, 0.313] | 0.1821 [0.165, 0.199] | 0.3401 [0.308, 0.371] | 0.1665 [0.147, 0.186] | 0.7912 [0.744, 0.835] | 0.7112 | 0.7505 [0.746, 0.755] |
| Collaborative Filtering | 0.2698 [0.243, 0.295] | 0.1564 [0.140, 0.174] | 0.3001 [0.269, 0.330] | 0.1584 [0.139, 0.179] | 0.7692 [0.718, 0.817] | 0.8556 | 0.7920 [0.788, 0.796] |
| Content-Based | 0.2360 [0.212, 0.260] | 0.1429 [0.129, 0.158] | 0.2919 [0.260, 0.325] | 0.1301 [0.112, 0.149] | 0.7656 [0.718, 0.813] | 0.9091 | 0.6643 [0.654, 0.674] |
| Fatoração de Matrizes (ALS) | 0.2071 [0.184, 0.229] | 0.1359 [0.121, 0.151] | 0.2616 [0.232, 0.289] | 0.1061 [0.091, 0.122] | 0.7289 [0.674, 0.777] | 0.9305 | 0.7956 [0.791, 0.800] |
| Bayesiano | 0.1848 [0.161, 0.209] | 0.1051 [0.091, 0.119] | 0.2015 [0.174, 0.229] | 0.1027 [0.086, 0.121] | 0.6081 [0.553, 0.667] | 0.9786 | 0.6080 [0.598, 0.617] |
| Popularidade | 0.1390 [0.120, 0.158] | 0.0982 [0.084, 0.112] | 0.1523 [0.129, 0.176] | 0.0671 [0.056, 0.078] | 0.5531 [0.495, 0.608] | 0.3048 | 0.7675 [0.763, 0.772] |
| Aleatório | 0.0593 [0.046, 0.073] | 0.0381 [0.031, 0.045] | 0.0694 [0.054, 0.086] | 0.0267 [0.019, 0.035] | 0.3077 [0.253, 0.363] | 1.0000 | 0.8205 [0.817, 0.824] |

### K = 5
| Estratégia | ndcg | precision | recall |
|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.2880 [0.257, 0.318] | 0.2337 [0.207, 0.259] | 0.2301 [0.201, 0.259] |
| Adaptativo (gating treinado) | 0.2800 [0.249, 0.312] | 0.2286 [0.203, 0.254] | 0.2247 [0.197, 0.253] |
| Adaptativo (regras) | 0.2700 [0.239, 0.300] | 0.2256 [0.201, 0.250] | 0.2228 [0.195, 0.251] |
| Híbrido pesos iguais | 0.2567 [0.228, 0.287] | 0.2139 [0.190, 0.239] | 0.2072 [0.182, 0.235] |
| Supervisionado | 0.2511 [0.223, 0.278] | 0.2110 [0.188, 0.235] | 0.2000 [0.173, 0.226] |
| Collaborative Filtering | 0.2493 [0.220, 0.277] | 0.2000 [0.176, 0.223] | 0.1987 [0.173, 0.226] |
| Content-Based | 0.2034 [0.178, 0.228] | 0.1678 [0.148, 0.188] | 0.1886 [0.162, 0.217] |
| Bayesiano | 0.1697 [0.143, 0.196] | 0.1297 [0.111, 0.150] | 0.1241 [0.103, 0.147] |
| Fatoração de Matrizes (ALS) | 0.1673 [0.143, 0.191] | 0.1458 [0.125, 0.167] | 0.1432 [0.121, 0.167] |
| Popularidade | 0.1236 [0.103, 0.144] | 0.1092 [0.092, 0.126] | 0.0873 [0.070, 0.106] |
| Aleatório | 0.0475 [0.034, 0.061] | 0.0403 [0.030, 0.051] | 0.0370 [0.026, 0.048] |

### K = 20
| Estratégia | ndcg | precision | recall |
|---|---|---|---|
| Adaptativo (gating treinado) | 0.3862 [0.359, 0.412] | 0.1498 [0.137, 0.162] | 0.5508 [0.518, 0.582] |
| Híbrido pesos fixos (ajustados) | 0.3745 [0.348, 0.400] | 0.1427 [0.130, 0.155] | 0.5145 [0.482, 0.544] |
| Adaptativo (regras) | 0.3654 [0.340, 0.391] | 0.1418 [0.130, 0.153] | 0.5222 [0.490, 0.555] |
| Supervisionado | 0.3615 [0.337, 0.385] | 0.1458 [0.134, 0.158] | 0.5388 [0.505, 0.572] |
| Híbrido pesos iguais | 0.3593 [0.334, 0.385] | 0.1427 [0.131, 0.155] | 0.5225 [0.490, 0.555] |
| Collaborative Filtering | 0.3273 [0.301, 0.353] | 0.1234 [0.112, 0.135] | 0.4529 [0.420, 0.485] |
| Content-Based | 0.2901 [0.267, 0.314] | 0.1128 [0.103, 0.123] | 0.4354 [0.401, 0.468] |
| Fatoração de Matrizes (ALS) | 0.2724 [0.250, 0.294] | 0.1167 [0.106, 0.128] | 0.4290 [0.397, 0.459] |
| Bayesiano | 0.2331 [0.210, 0.258] | 0.0875 [0.078, 0.097] | 0.3306 [0.298, 0.365] |
| Popularidade | 0.1898 [0.170, 0.209] | 0.0890 [0.079, 0.100] | 0.2910 [0.262, 0.320] |
| Aleatório | 0.0853 [0.072, 0.099] | 0.0381 [0.033, 0.043] | 0.1341 [0.114, 0.156] |

### Tempo de execução (segundos, ajuste + scoring no estágio de teste)
- Popularidade: 0.01s
- Content-Based: 1.00s
- Collaborative Filtering: 0.02s
- Fatoração de Matrizes (ALS): 0.28s
- Supervisionado: 0.32s
- Bayesiano: 0.38s
- Híbrido pesos iguais: 13.33s
- Híbrido pesos fixos (ajustados): 13.33s
- Adaptativo (regras): 13.33s
- Adaptativo (gating treinado): 13.33s

## 3. Comparações pareadas (ΔNDCG@10, bootstrap pareado)

| A | B | Δ (A−B) | IC95% | P(Δ ≤ 0) |
|---|---|---|---|---|
| Adaptativo (gating treinado) | Híbrido pesos fixos (ajustados) | -0.0027 | [-0.0190, +0.0144] | 0.619 |
| Adaptativo (regras) | Híbrido pesos fixos (ajustados) | -0.0137 | [-0.0285, +0.0028] | 0.946 |
| Adaptativo (gating treinado) | Adaptativo (regras) | +0.0110 | [-0.0015, +0.0233] | 0.039 |
| Adaptativo (gating treinado) | Supervisionado | +0.0246 | [+0.0060, +0.0435] | 0.004 |
| Adaptativo (regras) | Supervisionado | +0.0136 | [-0.0066, +0.0349] | 0.104 |
| Híbrido pesos fixos (ajustados) | Supervisionado | +0.0273 | [+0.0031, +0.0495] | 0.011 |
| Adaptativo (gating treinado) | Híbrido pesos iguais | +0.0236 | [+0.0107, +0.0367] | 0.001 |
| Supervisionado | Popularidade | +0.1497 | [+0.1227, +0.1768] | 0.000 |
| Collaborative Filtering | Fatoração de Matrizes (ALS) | +0.0628 | [+0.0418, +0.0852] | 0.000 |

## 4. Por disponibilidade de histórico (NDCG@10)

| Estratégia | histórico extenso (n=204) | histórico moderado (n=35) | pouco histórico (n=16) | sem histórico (n=18) |
|---|---|---|---|---|
| Adaptativo (gating treinado) | 0.3166 | 0.2859 | 0.4057 | 0.2472 |
| Híbrido pesos fixos (ajustados) | 0.3140 | 0.2975 | 0.4161 | 0.2848 |
| Adaptativo (regras) | 0.3058 | 0.2707 | 0.3908 | 0.2452 |
| Híbrido pesos iguais | 0.2858 | 0.2794 | 0.4062 | 0.2509 |
| Supervisionado | 0.2844 | 0.2836 | 0.3705 | 0.2746 |
| Collaborative Filtering | 0.2684 | 0.3132 | 0.3865 | 0.0976 |
| Content-Based | 0.2462 | 0.1919 | 0.2682 | 0.1764 |
| Fatoração de Matrizes (ALS) | 0.2055 | 0.2162 | 0.3308 | 0.0976 |
| Bayesiano | 0.1821 | 0.1874 | 0.2748 | 0.1305 |
| Popularidade | 0.1169 | 0.1631 | 0.2331 | 0.2585 |
| Aleatório | 0.0624 | 0.0429 | 0.0418 | 0.0711 |

### Por cenário detectado nos dados observados (NDCG@10)

| Estratégia | mudança de preferência (n=44) | padrão estável (n=149) | sazonal (n=62) | sem histórico (n=18) |
|---|---|---|---|---|
| Híbrido pesos fixos (ajustados) | 0.4403 | 0.2918 | 0.2948 | 0.2848 |
| Adaptativo (gating treinado) | 0.4281 | 0.3001 | 0.2826 | 0.2472 |
| Adaptativo (regras) | 0.4062 | 0.2886 | 0.2782 | 0.2452 |
| Híbrido pesos iguais | 0.3714 | 0.2775 | 0.2723 | 0.2509 |
| Collaborative Filtering | 0.3619 | 0.2583 | 0.2823 | 0.0976 |
| Content-Based | 0.2923 | 0.2309 | 0.2255 | 0.1764 |
| Supervisionado | 0.2511 | 0.3110 | 0.2657 | 0.2746 |
| Fatoração de Matrizes (ALS) | 0.2412 | 0.2152 | 0.1953 | 0.0976 |
| Popularidade | 0.1719 | 0.1281 | 0.1071 | 0.2585 |
| Bayesiano | 0.1139 | 0.1946 | 0.2273 | 0.1305 |
| Aleatório | 0.0632 | 0.0598 | 0.0519 | 0.0711 |

Cenários: *sem histórico* (0 pedidos no corte), *mudança de preferência* (JS > 0,15), *sazonal* (índice > 0,06), *padrão estável* (demais). Segmentos com n < 10 são indicativos, não conclusivos.

## 5. Seleção de especialistas pelo motor adaptativo

Pesos médios no teste:

| Gating | Content-Based | Collaborative Filtering | Supervisionado | Bayesiano | Popularidade |
|---|---|---|---|---|---|
| Híbrido pesos iguais | 0.20 | 0.18 | 0.20 | 0.20 | 0.20 |
| Híbrido pesos fixos (ajustados) | 0.12 | 0.54 | 0.34 | 0.00 | 0.00 |
| Adaptativo (regras) | 0.29 | 0.26 | 0.28 | 0.11 | 0.07 |
| Adaptativo (gating treinado) | 0.17 | 0.13 | 0.56 | 0.10 | 0.04 |

Gating treinável — seleção de complexidade por validação cruzada (5 dobras sobre clientes da validação;
empates de ±0,001 favorecem a variante mais simples):

| Variante | λ (L2 de θ) | NDCG@10 fora da dobra |
|---|---|---|
| regras | — | 0.2692 ± 0.0358 |
| viés | — | 0.2795 ± 0.0383 |
| viés+contexto | 0.03 | 0.2823 ± 0.0365 |
| viés+contexto | 0.1 | 0.2833 ± 0.0426 |
| viés+contexto | 0.3 | 0.2811 ± 0.0425 |
| viés+contexto | 1.0 | 0.2811 ± 0.0401 |
| viés+contexto | 3.0 | 0.2799 ± 0.0391 |
| viés+contexto | 10.0 | 0.2794 ± 0.0382 |

Variante escolhida: **viés+contexto** (λ = 0.03).

## 6. Modelo supervisionado — classificação no teste

| Algoritmo | ROC-AUC | PR-AUC | Log Loss | Brier | Precisão | Recall | Taxa base |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.7829 | 0.1449 | 0.1510 | 0.0376 | 0.149 | 0.404 | 0.0415 |
| random_forest | 0.8083 | 0.1625 | 0.1458 | 0.0370 | 0.203 | 0.290 | 0.0415 |
| lightgbm (selecionado) | 0.8115 | 0.1608 | 0.1460 | 0.0372 | 0.190 | 0.303 | 0.0415 |

Limiar de decisão definido no *holdout* do treino (máximo F1) e reaplicado no teste. Probabilidades calibradas por Platt.

## 7. Clusterização (corte do teste)

| k | Inertia | Silhouette | Estabilidade (ARI bootstrap) |
|---|---|---|---|
| 2 | 1929.9 | 0.248 | 0.955 |
| 3 | 1656.3 | 0.243 | 0.728 |
| 4 | 1395.6 | 0.268 | 0.850 |
| 5 | 1210.0 | 0.253 | 0.924 |
| 6 | 1103.5 | 0.252 | 0.917 |
| 7 | 1035.6 | 0.223 | 0.815 |
| 8 | 982.9 | 0.213 | 0.771 |
| 9 | 933.0 | 0.220 | 0.758 |
| 10 | 881.6 | 0.216 | 0.757 |

k escolhido = 5. Silhouette Ward = 0.242; ARI K-Means × Ward = 0.694.

## 8. Conclusões (geradas a partir dos números acima)

- Melhor NDCG@10 no teste: Híbrido pesos fixos (ajustados) = 0.3160 (IC95% 0.2870–0.3434).
- Adaptativo (gating treinado) − Híbrido pesos fixos (ajustados): ΔNDCG@10 = -0.0027 (IC95% -0.0190 a +0.0144) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (regras) − Híbrido pesos fixos (ajustados): ΔNDCG@10 = -0.0137 (IC95% -0.0285 a +0.0028) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (gating treinado) − Adaptativo (regras): ΔNDCG@10 = +0.0110 (IC95% -0.0015 a +0.0233) → diferença NÃO conclusiva (IC95% inclui zero).
- Adaptativo (gating treinado) − Supervisionado: ΔNDCG@10 = +0.0246 (IC95% +0.0060 a +0.0435) → ganho estatisticamente consistente (IC95% acima de zero).
- Adaptativo (regras) − Supervisionado: ΔNDCG@10 = +0.0136 (IC95% -0.0066 a +0.0349) → diferença NÃO conclusiva (IC95% inclui zero).
- Híbrido pesos fixos (ajustados) − Supervisionado: ΔNDCG@10 = +0.0273 (IC95% +0.0031 a +0.0495) → ganho estatisticamente consistente (IC95% acima de zero).
- Adaptativo (gating treinado) − Híbrido pesos iguais: ΔNDCG@10 = +0.0236 (IC95% +0.0107 a +0.0367) → ganho estatisticamente consistente (IC95% acima de zero).
- Supervisionado − Popularidade: ΔNDCG@10 = +0.1497 (IC95% +0.1227 a +0.1768) → ganho estatisticamente consistente (IC95% acima de zero).
- Collaborative Filtering − Fatoração de Matrizes (ALS): ΔNDCG@10 = +0.0628 (IC95% +0.0418 a +0.0852) → ganho estatisticamente consistente (IC95% acima de zero).
- Adaptativo (regras): o especialista de maior peso coincide com o melhor especialista do cliente em 23% dos casos (oráculo NDCG@10 = 0.4319).
- Adaptativo (gating treinado): o especialista de maior peso coincide com o melhor especialista do cliente em 34% dos casos (oráculo NDCG@10 = 0.4319).

## 9. Limitações e interpretação

- Os dados são sintéticos: o gerador define a utilidade latente; ganhos aqui não garantem ganhos em dados reais.
- Clientes avaliados: 273 no teste e 239 na validação
  (onde o gating é aprendido). A largura dos IC95% acima indica a menor diferença que este tamanho de amostra
  consegue distinguir do ruído.
- O gating treinável é regularizado em direção às regras e só substitui as regras quando vence na validação
  cruzada; quando isso não ocorre, ele reproduz exatamente o gating por regras.
- O especialista bayesiano opera no nível de categoria (por desenho): é útil como sinal de resposta a
  recomendações, mas não discrimina produtos dentro da categoria.
