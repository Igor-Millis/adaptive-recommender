# Manual do Usuário

> Projeto de fim de semana, feito para aprender e se divertir com sistemas de recomendação.

## Como abrir

Dê dois cliques em **`run.bat`** (no macOS/Linux, execute `./run.sh`). Na primeira execução ele cria o ambiente, instala as dependências,
gera os dados, treina os modelos e roda a avaliação (alguns minutos). Depois, a aplicação abre no
navegador em `http://localhost:8501`. Para fechar, feche a janela do terminal.

## Navegação

O menu lateral tem cinco páginas. Selos indicam a origem de cada informação:
**CADASTRO** (informado pelo lojista), **OBSERVADO** (calculado das compras) e **INFERIDO** (estimado por um modelo).

### 1. Dashboard Geral
Indicadores da operação (clientes, produtos, pedidos, faturamento, ticket médio), evolução mensal,
categorias mais vendidas, clientes por segmento e o mapa de perfil de consumo. Use os filtros de
período, categoria e segmento no topo; em "Destacar segmento" você realça um grupo no gráfico de dispersão.

### 2. Inteligência dos Clientes
Busque pelo nome, cidade ou segmento e clique numa linha da tabela. Você verá:
- os três blocos de dados (cadastro, observado, inferido);
- **Histórico de compras**: ticket ao longo do tempo, intervalo entre pedidos e todos os itens;
- **Preferências e incerteza**: participação real por categoria e a estimativa bayesiana de interesse
  com intervalo — quanto mais largo, menos certeza (pouca evidência);
- **Segmento e similares**: o cluster do cliente e os lojistas com padrão de compra mais parecido;
- **Evolução bayesiana**: como a estimativa mudou a cada interação.

### 3. Central de Recomendações
Escolha o cliente, a estratégia e o número de produtos e clique em **Gerar recomendações**.
- *Estratégia*: "Adaptativo" usa pesos diferentes para cada cliente; "Pesos fixos/iguais" são as referências.
- *Regras de ranking*: excluir produtos já comprados (padrão) ou permitir recompra antiga; diversificação
  (valores abaixo de 1 trazem mais variedade); peso da margem (0 = só interesse do cliente); limite por categoria.
- **Recomendações**: tabela com relevância, probabilidade de compra em 92 dias, preço e estoque; abra cada
  produto para ler os motivos e a contribuição de cada modelo.
- **Como o sistema chegou a essa recomendação**: pesos dos modelos, justificativa em linguagem natural,
  modelos descartados, nível de confiança e os sinais do cliente considerados.
- **Comparar estratégias**: listas de cada modelo isolado lado a lado e quanto elas se sobrepõem.

### 4. Simulador de Interações
Simula a reação do lojista. Escolha o cliente, um produto da lista e a interação:
- **Gostei** (interesse), **Clique**, **Adicionar ao carrinho**, **Comprar** (cria um pedido de verdade e baixa
  o estoque), **Não tenho interesse** (rejeição) ou **Ignorar** (apenas visualizou — evidência fraca).

Ao registrar, o sistema mostra o que de fato recalculou: a distribuição bayesiana antes/depois, a evolução das
estimativas, os pesos dos modelos e as recomendações antes × depois. Uma única interação normalmente muda
pouco os pesos — isso é esperado.

> As interações do simulador ficam gravadas no banco (origem "app"). Para voltar ao estado original,
> execute `python scripts/initialize_database.py --reset` e depois `python scripts/train_models.py`.

### 5. Avaliação dos Modelos (área técnica)
Resultados do experimento temporal: métricas de ranking com intervalos de confiança, comparações pareadas
com veredito estatístico, resultados por segmento, análise do motor adaptativo, métricas de classificação e
calibração do modelo supervisionado, diagnósticos de clusterização e a aba **Operações**, com versões dos
modelos, botões de **retreinamento** e de **reavaliação** e o log de eventos.

## Linha de comando (opcional)

```bash
python scripts/generate_data.py            # regera os CSVs sintéticos
python scripts/initialize_database.py      # cria/carrega o banco (use --reset para recriar)
python scripts/train_models.py             # retreinamento controlado (nova versão de cada modelo)
python scripts/run_evaluation.py           # experimento completo + docs/MODEL_EVALUATION.md
python -m pytest                           # testes automatizados
python scripts/download_real_data.py       # opcional: baixa o dataset real Online Retail II (~45 MB)
python scripts/run_real_data_experiment.py # opcional: mesmo experimento em dados reais
```
Antes, ative o ambiente: `.venv\Scripts\activate`.
