# Fundamentos Matemáticos e Estatísticos

Este documento descreve, para cada componente do Adaptive Recommender, **objetivo, hipóteses, notação,
equações, interpretação, limitações e onde está implementado**. As equações correspondem ao código — os
caminhos de arquivo indicam a implementação exata.

---

## 0. Notação geral

| Símbolo | Significado |
|---|---|
| $\mathcal{C}$, $c$ | conjunto de clientes (lojistas), cliente |
| $\mathcal{P}$, $p$ | catálogo de produtos, produto |
| $t$ | instante de corte (*as of*); só dados com data $< t$ são usados |
| $H$ | horizonte de previsão/avaliação (92 dias) |
| $\mathcal{O}_c(t)$ | itens de pedido do cliente $c$ antes de $t$: tuplas $(p, q, \tau, \text{preço})$ |
| $\mathcal{I}_c(t)$ | interações do cliente com recomendações antes de $t$ |
| $k \in \{1..K\}$ | categorias ($K = 9$) |
| $s_e(c,p)$ | score bruto do especialista $e$ |
| $\tilde s_e(c,p)$ | score normalizado em $[0,1]$ |
| $w_e(c)$ | peso do especialista $e$ para o cliente $c$ |
| $\mathcal{A}_c(t)$ | candidatos elegíveis de $c$ em $t$ |

---

## 1. Representação matemática dos dados

**Objetivo.** Transformar registros operacionais em objetos matemáticos estáveis e livres de vazamento.

**Hipóteses.** (i) Atributos cadastrais de clientes e de catálogo são estáticos (conhecidos no cadastro);
(ii) pedidos e interações são eventos datados; (iii) a disponibilidade de um produto em $t$ é
$\mathbb{1}[\text{lançamento}_p \le t]$.

**Equações.** Um *snapshot* é a restrição temporal
$$\mathcal{D}(t) = \{\text{cadastros}\} \cup \{o \in \text{pedidos}: \tau_o < t\} \cup \{i \in \text{interações}: \tau_i < t\}.$$
Toda feature é uma função $f(\mathcal{D}(t))$; todo rótulo é função de $\mathcal{D}(t+H) \setminus \mathcal{D}(t)$.

**Interpretação.** É a formalização de "o que o sistema sabia no momento da recomendação".

**Limitações.** Estoque é um estado atual (não há série histórica), por isso não é usado na avaliação
histórica — usá-lo seria vazamento.

**Aplicação.** `src/preprocessing/snapshot.py` (`DataSnapshot` valida via `validate_no_future`).

### Atributos observados × inferidos
- **Observados (histórico)**: contagens e médias (nº de pedidos, ticket médio $\bar v_c = \frac{1}{|O_c|}\sum_o v_o$,
  frequência $= |O_c| / \text{meses ativos}$, recência).
- **Inferidos**: dependem de um modelo/hipótese:
  - índice de preço $\pi_c = \frac{\sum_i q_i \log(\text{preço}_i/\text{mediana}_{k(i)})}{\sum_i q_i}$ (negativo ⇒ compra abaixo da mediana da categoria, proxy de sensibilidade a preço);
  - diversidade $H_c = -\sum_k \sigma_{ck}\log\sigma_{ck} / \log K$, com $\sigma_{ck}$ = participação da categoria nas unidades;
  - sazonalidade pela 1ª harmônica circular: $\bar R = \frac{|\sum_m n_m e^{i 2\pi m/12}|}{n}$, corrigida pelo viés de amostra
    $S_c = \max\!\left(0, \frac{n\bar R^2 - 1}{n-1}\right)$ (sob uniformidade $E[n\bar R^2] \approx 1$);
  - mudança de preferência: divergência de Jensen-Shannon entre as participações por categoria dos últimos 120 dias e do período anterior,
    $\mathrm{JS}(P\|Q) = \tfrac12 \mathrm{KL}(P\|M) + \tfrac12\mathrm{KL}(Q\|M)$, $M = \tfrac12(P+Q)$, em bits ($\in[0,1]$);
  - consistência: cosseno entre as participações na 1ª e na 2ª metade dos pedidos.

Implementação: `src/preprocessing/feature_engineering.py`, `src/preprocessing/transformations.py`.

---

## 2. Construção da matriz cliente-produto

**Objetivo.** Codificar feedback implícito (compras e engajamentos) com intensidade e recência.

**Hipóteses.** Quantidade tem retorno decrescente; compras antigas informam menos; engajamento sem compra
é evidência mais fraca que compra.

**Equação.**
$$R_{cp} = \sum_{(p,q,\tau)\in\mathcal{O}_c} \log(1+q)\, 2^{-(t-\tau)/h} \;+\; w_e \sum_{i \in \mathcal{I}^+_c,\, p_i = p} 2^{-(t-\tau_i)/h},$$
com $h = 365$ dias e $w_e = 0{,}3$.

**Interpretação.** $R_{cp}=0$ significa *ausência de observação*, não desinteresse.

**Limitações.** Densidade ≈ 26% (300 × 200): pouco esparsa para padrões de produção, mas com clientes
sem histórico (linhas nulas).

**Aplicação.** `src/recommenders/collaborative.py::implicit_matrix`.

---

## 3. Similaridade de cosseno

**Objetivo.** Medir semelhança de direção entre vetores, invariante à escala.

$$\cos(u, v) = \frac{u^\top v}{\|u\|\,\|v\|} \in [-1,1] \quad (\in [0,1] \text{ para vetores não negativos}).$$

**Shrinkage** (filtragem colaborativa): $s'(c,v) = \cos(R_c, R_v)\cdot \frac{n_{cv}}{n_{cv}+\lambda}$,
$n_{cv}$ = nº de produtos em comum, $\lambda = 5$. Reduz similaridades altas obtidas com pouca sobreposição
(estimativas de alta variância).

**Limitações.** Não considera popularidade do item (itens populares inflam similaridades).

**Aplicação.** CF user-kNN, perfil Content-Based, similaridade produto-produto (MMR e diversidade).

---

## 4. TF-IDF

**Objetivo.** Transformar texto descritivo em vetores que destacam termos discriminantes.

$$\mathrm{tfidf}(w, d) = \mathrm{tf}(w,d)\cdot \left(\log\frac{1+N}{1+\mathrm{df}(w)} + 1\right),$$
seguido de normalização L2 (implementação do scikit-learn, `min_df = 2`, acentos removidos).

**Representação do produto (Content-Based).** Concatenação de blocos, cada bloco $b$ normalizado em L2 e
multiplicado por $\sqrt{\omega_b}$:
$$x_p = \big[\sqrt{\omega_{\text{cat}}}\,\hat e_{\text{cat}},\ \sqrt{\omega_{\text{sub}}}\,\hat e_{\text{sub}},\ \dots,\ \sqrt{\omega_{\text{txt}}}\,\widehat{\mathrm{tfidf}}\big],\qquad x_p \leftarrow x_p/\|x_p\|.$$
Assim $x_p^\top x_{p'} \propto \sum_b \omega_b \cos_b(p,p')$: os pesos $\omega_b$ (configuração `CB.block_weights`)
controlam diretamente a importância de cada atributo.

**Perfil do cliente (Rocchio ponderado).**
$$u_c = \sum_{(p,q,\tau)} \log(1+q)\,2^{-(t-\tau)/h_c}\, x_p \;+\; \sum_{i\in\mathcal{I}_c} \beta_{\text{tipo}(i)}\,2^{-(t-\tau_i)/h_c}\,x_{p_i},$$
$\beta$: clique 0,25; interesse 0,5; carrinho 0,7; **rejeição −0,7**. Meia-vida $h_c = 180$ dias, reduzida para
60 dias se $\mathrm{JS}_c > 0{,}15$ (mudança de preferência). Cold start: média dos $x_p$ das categorias e faixa de
preço declaradas no cadastro.

**Score e explicação exata.** $s_{CB}(c,p) = \cos(u_c, x_p) = \sum_j \frac{u_{cj}}{\|u_c\|} x_{pj}$; cada termo
$j$ é a contribuição do atributo $j$ — a explicação soma exatamente o score.

**Aplicação.** `src/recommenders/content_based.py`.

---

## 5. Algoritmos de clusterização

**Objetivo.** Descobrir segmentos interpretáveis usados como contexto (prior bayesiano, features do
supervisionado, explicações) — **não** como recomendador.

**Variáveis.** log(ticket médio), log(frequência), diversidade, log(preço unitário médio), índice de preço,
log(recência), sazonalidade e as 9 participações por categoria; padronização $z = (x-\mu)/\sigma$; o bloco
composicional de participações é multiplicado por $\sqrt{3/9}$ para pesar como ~3 variáveis.

**K-Means.** Minimiza a inércia $J = \sum_{c} \|z_c - \mu_{g(c)}\|^2$ (Lloyd, `n_init = 20`).

**Hierárquico (Ward).** Une, a cada passo, o par de grupos que minimiza o aumento da soma de quadrados
intra-grupo: $\Delta(A,B) = \frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|^2$.

**Escolha de $k$.**
- Silhouette: $s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$, média sobre clientes;
- Estabilidade: média do *Adjusted Rand Index* entre a partição de referência e 30 re-ajustes em amostras
  bootstrap (cada re-ajuste rotula todos os clientes);
- Inércia: diagnóstico de cotovelo.
Critério: $k^* = \arg\max_{k\ge 3} \text{silhouette}(k)\cdot\text{estabilidade}(k)$.

**Interpretabilidade.** Rótulos derivados de z-scores dos centróides (ex.: participação dominante > 28% com
*lift* > 1,6 ⇒ "Especialistas em ⟨categoria⟩"; z(ticket) > 0,7 ⇒ "alto ticket"; z(índice de preço) < −0,7 ⇒
"sensíveis a preço"). Ids de cluster são alinhados entre retreinos pelo algoritmo húngaro sobre distâncias
entre centróides.

**Limitações.** ~280 clientes clusterizáveis (k testado de 2 a 10); silhouette ≈ 0,25 indica estrutura moderada. Clientes com < 3
pedidos não são clusterizados (rótulo −1).

**Aplicação.** `src/segmentation/clustering.py`, `src/segmentation/cluster_analysis.py`.

---

## 6. Regressão logística

**Objetivo.** Estimar $P(Y=1\mid x)$, onde $Y$ = "cliente comprará o produto em $[t, t+H)$".

**Modelo.** $P(Y=1\mid x) = \sigma(\beta_0 + \beta^\top x)$, $\sigma(z) = 1/(1+e^{-z})$; estimação por máxima
verossimilhança penalizada (L2, $C = 0{,}5$) com pesos amostrais:
$$\hat\beta = \arg\min_\beta -\sum_i w_i\left[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\right] + \tfrac{1}{2C}\|\beta\|^2.$$

**Hipóteses.** Log-odds linear nas features (padronizadas); observações condicionalmente independentes
(aproximação: pares de um mesmo cliente são correlacionados).

**Explicação.** Contribuição da feature $j$: $\beta_j z_j$ (escala log-odds).

### Construção do dataset
- Snapshots mensais $t_1 < \dots < t_m$ com $t_j + H \le$ corte; features em $\mathcal{D}(t_j)$; rótulo na janela seguinte.
- **Candidatos**: produtos lançados e ainda não comprados (tarefa de descoberta).
- **Negativos não-definitivos**: (a) cada par é rotulado *por janela* — pode ser negativo em $t_j$ e positivo em
  $t_{j+1}$; (b) só entram clientes **ativos** na janela (≥ 1 pedido: houve oportunidade de compra);
  (c) exposição prévia (impressões, engajamento, rejeição do produto) é feature, permitindo ao modelo
  distinguir "não viu" de "viu e não quis".
- **Desbalanceamento** (~4–5% positivos): negativos subamostrados com taxa $r = 0{,}35$ e peso $1/r$
  (estimador de Horvitz-Thompson), o que preserva a esperança da log-verossimilhança e portanto a calibração.

**Aplicação.** `src/recommenders/supervised.py` (`build_labeled_snapshot`, `pair_features`).

---

## 7. Árvores e modelos de boosting

**Random Forest.** Média de $B = 300$ árvores treinadas em amostras bootstrap com subconjunto aleatório de
variáveis por nó ($\sqrt{d}$): $\hat p(x) = \frac1B\sum_b T_b(x)$. Reduz variância por decorrelação.

**Gradient Boosting (LightGBM).** Modelo aditivo em log-odds
$F_M(x) = \sum_{m=1}^{M} \eta\, f_m(x)$, cada árvore $f_m$ ajustada ao gradiente/hessiana da log-loss
(aproximação de Newton de 2ª ordem), com $\eta = 0{,}03$, 15 folhas, regularização L2 e subamostragem.
Explicação por TreeSHAP (`pred_contrib`), que decompõe exatamente $F(x) = \phi_0 + \sum_j \phi_j(x)$.

**Seleção.** O algoritmo com maior PR-AUC num *holdout temporal* (último snapshot, posterior aos de treino)
é selecionado. PR-AUC é preferida à ROC-AUC sob desbalanceamento.

---

## 8. Estimativas de probabilidade e calibração

**Objetivo.** Scores devem ser probabilidades interpretáveis ($P = 0{,}2$ ⇒ ~20% de compras).

**Calibração de Platt sobre o logit.** $\tilde p = \sigma(a\cdot \mathrm{logit}(\hat p) + b)$, com $(a,b)$ por máxima
verossimilhança no holdout. Avaliação: Brier $= \frac1n\sum(\tilde p_i - y_i)^2$, log-loss e diagrama de
confiabilidade (decis de $\tilde p$ × frequência observada).

**Limitação.** O calibrador é ajustado em um único snapshot de holdout; mudanças de taxa-base entre períodos
(sazonalidade) podem descalibrar.

---

## 9. Distribuição Beta

$$\theta \sim \mathrm{Beta}(\alpha,\beta),\quad f(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\quad
E[\theta] = \frac{\alpha}{\alpha+\beta},\quad \mathrm{Var}[\theta] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.$$
$\kappa = \alpha+\beta$ funciona como "tamanho de amostra equivalente" do prior; $\mu = \alpha/\kappa$ é a média.

---

## 10. Atualização Beta-Binomial (modelo de dois estágios)

**Objetivo.** Estimar, com incerteza, a resposta de cada cliente a recomendações de cada categoria, e
atualizar a estimativa a cada nova interação.

**Granularidade.** Cliente × categoria (2.700 células) — **não** cliente × produto (60.000 células com,
em média, menos de 2 exposições cada: estimativas dominadas pelo prior).

**Modelo em funil.**
$$\theta^{I}_{ck} = P(\text{engajamento positivo}\mid \text{recomendação de }k),\qquad
\theta^{C}_{ck} = P(\text{compra}\mid\text{engajamento}),\qquad
P(\text{compra}\mid\text{rec.}) = \theta^I_{ck}\,\theta^C_{ck}.$$

**Desfecho resolvido.** Cada (sessão, produto) recebe o desfecho mais forte observado:
compra ≻ carrinho ≻ interesse ≻ clique ≻ rejeição ≻ impressão.

**Evidência por desfecho** (`outcome_evidence`):

| desfecho | estágio interesse $(s, f)$ | estágio conversão $(s, f)$ |
|---|---|---|
| compra | (1, 0) | (1, 0) |
| carrinho sem compra | (1, 0) | (0, 0,5) |
| clique / interesse | (1, 0) | (0, 1) |
| rejeição explícita | (0, 1) | — |
| impressão sem resposta | (0, η = 0,25) | — |

Consequências exigidas pela especificação:
1. **Compra é evidência mais forte que clique**: incrementa sucessos nos dois estágios.
2. **Impressão sem interação não é fracasso de conversão**: nunca entra no estágio de conversão; no estágio de
   interesse vale apenas η = 0,25 de fracasso (ambígua: o item pode não ter sido visto).
3. **Ausência de clique ≠ rejeição**: rejeição explícita vale 1; impressão, 0,25.

**Hipótese / justificativa da evidência fracionária.** Contagens fracionárias correspondem a uma
verossimilhança *temperada* (power likelihood): $L(\theta)^{w} = \theta^{ws}(1-\theta)^{wf}$. A conjugação
Beta se mantém: $\theta\mid\cdot \sim \mathrm{Beta}(\alpha + \sum w s, \beta + \sum w f)$. O peso $w < 1$ expressa
a confiança de que o evento seja de fato um "fracasso" do mesmo processo binomial.

**Atualização (conjugada).**
$$\theta\mid\text{dados} \sim \mathrm{Beta}(\alpha_0 + s, \beta_0 + f),\qquad
E[\theta\mid\text{dados}] = \frac{\alpha_0 + s}{\alpha_0+\beta_0+s+f}.$$

**Atualização incremental consistente.** Quando chega um evento novo numa sessão, aplica-se
$\Delta = \mathrm{ev}(\text{desfecho novo}) - \mathrm{ev}(\text{desfecho anterior})$. Por telescopia,
$\sum \Delta = \mathrm{ev}(\text{desfecho final})$ — o estado incremental é idêntico ao recálculo em lote
(verificado para todas as sequências de até 3 eventos em `tests/test_bayesian.py` e no teste de ponta a ponta).

**Aplicação.** `src/recommenders/bayesian.py`, `src/learning/bayesian_updates.py`, `src/learning/feedback.py`.

---

## 11. Inferência posterior e prior hierárquico (Bayes empírico)

**Prior por categoria e estágio** — maximiza a verossimilhança marginal Beta-Binomial entre clientes:
$$(\hat\mu_k,\hat\kappa_k) = \arg\max_{\mu,\kappa} \sum_c \Big[\log B(\mu\kappa + s_{ck},\,(1-\mu)\kappa + f_{ck}) - \log B(\mu\kappa,(1-\mu)\kappa)\Big],$$
com $\kappa \in [2, 60]$ (L-BFGS sobre $\mathrm{logit}\,\mu$ e $\log\kappa$).

**Média do segmento com leave-one-out e encolhimento** (o cliente não entra no próprio prior):
$$\mu_{gck} = \frac{S_{g,k}^{(-c)} + \lambda\hat\mu_k}{N_{g,k}^{(-c)} + \lambda},\quad \lambda = 20;\qquad
\alpha_0 = \mu_{gck}\hat\kappa_k,\ \beta_0 = (1-\mu_{gck})\hat\kappa_k.$$

**Intervalo de credibilidade** (90%): quantis $\alpha$-/2 e 1−$\alpha$/2 da Beta posterior.
Para $P(\text{compra}\mid\text{rec.}) = \theta^I\theta^C$ (posteriores independentes): média exata
$E[\theta^I]E[\theta^C]$; intervalo por Monte Carlo (4.000 amostras).

**Exploração (opcional).** Amostragem de Thompson: $\tilde\theta \sim$ posterior; disponível em
`thompson_scores`.

**Limitações.** (i) Independência entre estágios e entre categorias; (ii) não modela decaimento temporal
das evidências (preferências estacionárias dentro da janela); (iii) η e os pesos fracionários são
hiperparâmetros de modelagem, não estimados.

---

## 12. Construção dos scores

| Especialista | Score bruto $s_e(c,p)$ | Confiança $\gamma_e(c)\in[0,1]$ |
|---|---|---|
| Popularidade | $\sum_{c' \text{ compradores}} 2^{-\text{idade}/60} + 0{,}1\cdot\text{compradores}/\max$ | 1 |
| Content-Based | $\cos(u_c, x_p)$ | $1 - e^{-n_c/15}$ (0,15 se só cadastro) |
| Collaborative (user-kNN) | $\sum_{v\in N_k(c)} s'(c,v)\tilde R_{vp} / \sum_v s'(c,v)$ | média das $k$ maiores similaridades |
| Supervisionado | $\tilde p(c,p)$ calibrada | $0{,}35 + 0{,}65(1-e^{-n/10})$ |
| Bayesiano | $E[\theta^I_{ck(p)}]E[\theta^C_{ck(p)}] + 10^{-4}\,\text{pop}_p$ | $\overline{n^{\text{eff}}_{ck}/(n^{\text{eff}}_{ck}+\kappa)}$ |
| ALS (comparação) | $x_c^\top y_p$ | $1-e^{-n_c/20}$ |

O termo $10^{-4}\,\text{pop}_p$ do bayesiano só desempata produtos da mesma categoria (o modelo é, por
desenho, categórico).

**ALS implícito (Hu, Koren & Volinsky, 2008).** $\min \sum_{cp} C_{cp}(P_{cp} - x_c^\top y_p)^2 + \lambda(\sum\|x_c\|^2 + \sum\|y_p\|^2)$,
$P = \mathbb{1}[R>0]$, $C = 1 + \alpha R$; passos fechados $x_c = (Y^\top C^c Y + \lambda I)^{-1} Y^\top C^c P_c$.

---

## 13. Normalização

**Problema.** Escalas incomparáveis (cossenos ~0,3–0,9, probabilidades ~0,01–0,9, contagens).

**Normalização por percentil dentro dos candidatos do cliente**:
$$\tilde s_e(c,p) = \frac{\mathrm{rank}_{q\in\mathcal{A}_c}(s_e(c,q)) - 1}{|\mathcal{A}_c| - 1} \in [0,1],$$
empates com rank médio; especialista constante (sem informação) ⇒ $\tilde s = 0{,}5$ para todos (não altera a
ordem). Alternativa min-max implementada (`minmax_normalize`).

**Interpretação.** A combinação passa a ser uma *agregação de rankings ponderada* (estilo Borda ponderado),
robusta a caudas e escalas.

**Limitação.** Descarta a magnitude: um especialista "muito confiante" num item não se destaca mais que no
percentil. A confiança é tratada separadamente pelo gating.

---

## 14. Combinação dos especialistas (Mixture of Experts)

$$s(c,p) = \sum_{e\in\mathcal{E}} w_e(c)\,\tilde s_e(c,p),\qquad w_e(c)\ge 0,\quad \sum_e w_e(c) = 1.$$

As restrições são verificadas em tempo de execução (`validate_weights`). Contribuição exibida na interface:
$w_e(c)\tilde s_e(c,p)/s(c,p)$.

### Gating por regras
Confiabilidades $r_e(c) \in [0,1]$ por funções de saturação $\mathrm{sat}(x, a) = 1 - e^{-x/a}$:
- $r_{CB} = 0{,}35 + 0{,}45\,\mathrm{sat}(n^{\text{prod}}_c, 15) + 0{,}2\,\text{consistência}_c$ (0,15 se só cadastro);
- $r_{CF} = \mathrm{sat}(n^{\text{prod}}_c, 20)\cdot\min(1, \bar s_{N_k}/0{,}4)$ se $n^{\text{prod}}_c \ge 3$, senão 0;
- $r_{SUP} = 0{,}4 + 0{,}3\,\mathrm{sat}(n^{\text{ped}}_c,10) + 0{,}2\,\mathrm{sat}(n^{\text{imp}}_c,60) + 0{,}15\cdot\mathbb{1}[\mathrm{JS}_c>0{,}15]$;
- $r_{BAY} = 0{,}05 + 0{,}6\,\gamma_{BAY}(c)$;
- $r_{POP} = 0{,}05 + 0{,}75\,e^{-n^{\text{prod}}_c/5}$;
$$w_e(c) = r_e(c)/\textstyle\sum_{e'} r_{e'}(c).$$
Cada regra corresponde a um item da especificação (poucos pedidos ⇒ menos CF; histórico consistente ⇒ mais CB;
muitos vizinhos ⇒ mais CF; muitas interações ⇒ mais supervisionado/bayesiano; mudança recente ⇒ mais
informação recente).

---

## 15. Otimização dos pesos (gating treinável)

**Modelo residual sobre as regras.**
$$z_e(c) = \log r_e(c) + b_e + \theta_e^\top\phi(c),\qquad w(c) = \mathrm{softmax}(z(c)),$$
$\phi(c)$ = 11 sinais padronizados (log pedidos, log produtos, log impressões, densidade, diversidade,
força dos vizinhos, log recência, deriva, consistência, evidência bayesiana, tem-histórico). Com
$b=\theta=0$ recupera exatamente as regras; $r_e = 0$ ⇒ $w_e = 0$ (indisponibilidade respeitada).

**Função objetivo (listwise, janela de validação).**
$$\mathcal{L}(b,\theta) = \frac{1}{|\mathcal{C}^+|}\sum_{c}\left[-\frac{1}{|P_c|}\sum_{p\in P_c}\log\frac{e^{s(c,p)/T}}{\sum_{q\in\mathcal{A}_c}e^{s(c,q)/T}}\right] + \frac{\lambda_b}{2}\|b\|^2 + \frac{\lambda}{2}\|\theta\|^2,$$
$T = 0{,}1$, $P_c$ = compras novas na janela. Gradiente analítico:
$\partial\mathcal{L}/\partial s_{cq} = (\pi_{cq} - y_{cq}/|P_c|)/T$;
$g_{ce} = \sum_q \partial\mathcal{L}/\partial s_{cq}\,\tilde s_{ceq}$;
$\partial\mathcal{L}/\partial z_{ce} = w_{ce}(g_{ce} - \sum_{e'}w_{ce'}g_{ce'})$ (Jacobiano do softmax);
otimização por L-BFGS (gradiente verificado por diferenças finitas nos testes).

**Prior e seleção de complexidade.** A penalidade L2 equivale a um prior gaussiano centrado nas regras.
Por validação cruzada (5 dobras **sobre clientes**, NDCG@10 fora da dobra) escolhe-se entre:
regras puras, só viés ($\theta \equiv 0$) e viés + contexto ($\lambda$ na grade). Empates (±0,001) favorecem
o modelo mais simples.

**Sem vazamento.** O gating avaliado no teste é aprendido **apenas** na janela de validação
(mai–jul/2026) com especialistas ajustados em dados < mai/2026. Em produção, ele é reaprendido na janela
completa mais recente $[t-H, t)$ com especialistas ajustados em $< t-H$.

**Pesos fixos (baseline justo).** $w^\star = \arg\max_{w\in\Delta,\ \text{passo }0{,}1}\ \overline{\mathrm{NDCG}@10}$ na validação
(1.001 combinações), os mesmos para todos os clientes.

**Aplicação.** `src/adaptive/gating.py`, `src/adaptive/model_selection.py`.

---

## 16. Função de ranking

1. **Elegibilidade**: $\mathcal{A}_c = \{p: \text{lançado} \wedge \text{estoque}>0 \wedge \neg\text{comprado (regra)} \wedge \neg\text{rejeitado há < 180 dias}\}$.
2. **Fadiga**: $\phi_{cp} = 0{,}15$ se o produto foi exibido ≥ 3 vezes sem resposta nos últimos 60 dias.
3. **Objetivo comercial separado**: $r(c,p) = (1-\mu)\,s(c,p)(1-\phi_{cp}) + \mu\,\widetilde{\text{margem}}_p$, $\mu$ configurável
   (padrão 0 — relevância pura; a relevância exibida nunca inclui margem).
4. **Diversificação (MMR)**: $p^\star = \arg\max_{p\notin L}\ \lambda\,\tilde r(c,p) - (1-\lambda)\max_{q\in L}\cos(x_p,x_q)$,
   com limite opcional por categoria.

**Aplicação.** `src/ranking/*.py`, `src/services/recommendation_service.py`.

---

## 17. Métricas de avaliação

Relevância binária $\mathrm{rel}(c,p) = \mathbb{1}[p \text{ comprado em } [t,t+H) \wedge p\in\mathcal{A}_c(t)]$; macro-média sobre clientes com ≥ 1 relevante.

- $\mathrm{Precision@K} = |L_K\cap R|/K$; $\mathrm{Recall@K} = |L_K\cap R|/|R|$; $\mathrm{HitRate@K} = \mathbb 1[L_K\cap R\ne\emptyset]$.
- $\mathrm{DCG@K} = \sum_{i=1}^{K}\mathrm{rel}_i/\log_2(i+1)$; $\mathrm{NDCG@K} = \mathrm{DCG@K}/\mathrm{IDCG@K}$.
- $\mathrm{AP@K} = \frac{1}{\min(|R|,K)}\sum_{i\le K}\mathrm{Prec@}i\cdot\mathrm{rel}_i$; MAP = média.
- Cobertura $= |\bigcup_c L_K(c)| / |\mathcal{P}_{\text{lançados}}|$.
- Diversidade intra-lista $= 1 - \frac{1}{K(K-1)}\sum_{i\ne j}\cos(x_i,x_j)$.
- Classificação: ROC-AUC, PR-AUC (*average precision*), log-loss, Brier, precisão/recall no limiar de máximo F1
  definido no holdout.
- **Incerteza**: IC95% por bootstrap não paramétrico sobre clientes (2.000 reamostragens); comparações por
  bootstrap **pareado** da diferença por cliente, reportando IC e $P(\Delta\le 0)$.

**Limitação.** Com ~270 clientes avaliados, os IC95% de NDCG@10 têm largura ≈ ±0,015 a ±0,03 (com 50 clientes, ≈ ±0,07);
o relatório declara "não conclusivo" sempre que o IC inclui zero.

**Aplicação.** `src/evaluation/metrics.py`, `src/evaluation/comparison.py`, `src/evaluation/experiments.py`.
