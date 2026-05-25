# TP1 NLP - Assistente de Rascunho com Verbalized Sampling

Pipeline experimental em Python para comparar geracao direta (`baseline`) contra Verbalized Sampling (`VS`) em mensagens cotidianas. A entrega oficial usa **LM Studio local** como backend de geracao, com 3 modelos locais e 3 condicoes experimentais por modelo.

## Estrutura

```text
configs/              Configuracao oficial do experimento
data/                 Dataset e templates de prompt versionados
scripts/              CLIs de execucao, validacao e analise
src/tp1_nlp/          Codigo modular do pipeline
results/runs/         Artefatos brutos de cada execucao
results/final/        Artefatos finais selecionados para entrega
tests/                Testes com pytest
```

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Perplexity usa `transformers` e `torch`, que estao comentados em `requirements.txt` por serem dependencias pesadas. Sem essas dependencias, use `--skip-perplexity`; as demais metricas continuam sendo calculadas.

## Modelos Locais

O experimento oficial espera o servidor local de chat do LM Studio ativo em:

```text
http://localhost:1234/v1
```

Modelos configurados em `configs/experiment.lmstudio.yaml`:

```text
lmstudio_qwen35_9b      -> qwen3.5-9b
lmstudio_gemma_e4b      -> gemma-4-e4b-it-mlx
lmstudio_ministral_14b  -> ministral-3-14b-instruct-2512
```

Antes de rodar o experimento, abra o LM Studio, carregue os modelos e inicie o servidor local.

## Validacao

Validar estrutura, dataset e prompts:

```bash
python3 scripts/validate_project.py
```

Validar se o LM Studio esta ativo e expondo os 3 modelos:

```bash
python3 scripts/check_lmstudio.py --config configs/experiment.lmstudio.yaml
```

Executar smoke test por modelo:

```bash
python3 scripts/smoke_lmstudio.py --model qwen
python3 scripts/smoke_lmstudio.py --model gemma
python3 scripts/smoke_lmstudio.py --model ministral
```

Cada smoke gera um `run_id` automatico e roda apenas a condicao `baseline` para o modelo escolhido.

## Execucao Oficial

Rodar o experimento completo:

```bash
python3 scripts/run_experiment.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_final
```

Cada execucao cria:

```text
results/runs/<run_id>/
  manifest.json
  run.log
  generations.jsonl
  raw_responses/
  figures/
```

O `manifest.json` registra dataset, hash do dataset, modelos, parametros, prompts, ambiente e caminhos dos arquivos gerados.

## Metricas e Graficos

Calcular metricas automaticas:

```bash
python3 scripts/compute_metrics.py --run-id lmstudio_final --skip-perplexity
```

Gerar graficos e exemplos:

```bash
python3 scripts/make_plots.py --run-id lmstudio_final
```

Saidas principais:

```text
results/runs/lmstudio_final/metrics_long.csv
results/runs/lmstudio_final/metrics_summary.csv
results/runs/lmstudio_final/figures/
results/runs/lmstudio_final/response_examples.csv
```

Metricas automaticas implementadas:

- `jaccard`: sobreposicao lexical media par-a-par; menor tende a indicar maior diversidade.
- `self_bleu`: repeticao entre respostas; menor tende a indicar maior diversidade.
- `distinct_1` e `distinct_2`: variedade lexical; maior tende a indicar maior diversidade.
- `tone_coverage`: cobertura dos tons esperados por mensagem.
- `perplexity`: opcional, com modelo avaliador fixo via `transformers`.

## Avaliacao Manual

Exportar planilha:

```bash
python3 scripts/export_manual_eval.py --run-id lmstudio_final
```

Depois de preencher notas de 1 a 5:

```bash
python3 scripts/import_manual_eval.py \
  --run-id lmstudio_final \
  --file results/runs/lmstudio_final/manual_eval_template.csv
```

Campos avaliados:

- utilidade;
- coerencia;
- fidelidade a mensagem original;
- adequacao ao tom pedido.

## Politica de Resultados

`results/runs/*` e ignorado por padrao para evitar versionar execucoes intermediarias, logs grandes e tentativas quebradas.

Para a entrega, use um run limpo chamado `lmstudio_final` e copie apenas os artefatos finais relevantes para:

```text
results/final/lmstudio_final/
```

Artefatos recomendados para entrega:

```text
manifest.json
generations.jsonl
metrics_long.csv
metrics_summary.csv
figures/
response_examples.csv
manual_eval_template.csv
manual_metrics_summary.csv, se houver avaliacao manual preenchida
```

Nao use runs antigos de debug como resultado oficial.

## Decisao sobre Baseline

O baseline usa prompt direto, mas e repetido `baseline_repetitions` vezes por mensagem. Isso preserva a condicao baseline e permite calcular metricas de diversidade como Jaccard par-a-par e Self-BLEU de forma comparavel ao VS, que gera 5 alternativas.

## Testes

```bash
pytest
```

Os testes automatizados usam `FakeLLMClient` internamente para validar o pipeline sem depender do LM Studio. O experimento oficial, entretanto, usa somente os modelos locais configurados no LM Studio.
