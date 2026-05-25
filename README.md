# TP1 NLP - Assistente de Rascunho com Verbalized Sampling

Pipeline experimental em Python para comparar geracao direta (`baseline`) contra Verbalized Sampling (`VS`) em mensagens cotidianas. O projeto segue a especificacao do TP1: 20 a 30 mensagens, 3 LLMs, 3 condicoes por LLM, registro de prompts/parametros/ambiente, metricas automaticas, avaliacao manual por CSV e graficos comparativos.

## Estrutura

```text
configs/              Configuracoes de experimento e exemplos de modelos reais
data/                 Dataset e templates de prompt versionados
scripts/              CLIs de execucao
src/tp1_nlp/          Codigo modular do pipeline
results/runs/         Artefatos de cada execucao
tests/                Testes com pytest
```

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Perplexity usa `transformers` e `torch`, que estao comentados em `requirements.txt` por serem dependencias pesadas. Sem essas dependencias, use `--skip-perplexity`; as demais metricas continuam sendo calculadas.

## Validar o projeto

```bash
python scripts/validate_project.py
```

## Executar experimento

O arquivo `configs/experiment.yaml` vem com tres modelos `fake` para permitir execucao reprodutivel sem API paga:

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

Execucao parcial para smoke test:

```bash
python scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --only-model fake_model_a \
  --only-condition vs_intermediate
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

## Calcular metricas

```bash
python scripts/compute_metrics.py --run-id <run_id> --skip-perplexity
```

Saidas:

```text
results/runs/<run_id>/metrics_long.csv
results/runs/<run_id>/metrics_summary.csv
```

Metricas automaticas implementadas:

- `jaccard`: sobreposicao lexical media par-a-par; menor tende a indicar maior diversidade.
- `self_bleu`: repeticao entre respostas; menor tende a indicar maior diversidade.
- `distinct_1` e `distinct_2`: variedade lexical; maior tende a indicar maior diversidade.
- `tone_coverage`: cobertura dos tons esperados por mensagem.
- `perplexity`: opcional, com modelo avaliador fixo via `transformers`.

## Gerar graficos e exemplos

```bash
python scripts/make_plots.py --run-id <run_id>
```

Saidas em:

```text
results/runs/<run_id>/figures/
results/runs/<run_id>/response_examples.csv
```

## Avaliacao manual

Exportar planilha:

```bash
python scripts/export_manual_eval.py --run-id <run_id>
```

Depois de preencher notas de 1 a 5:

```bash
python scripts/import_manual_eval.py \
  --run-id <run_id> \
  --file results/runs/<run_id>/manual_eval_template.csv
```

Campos avaliados:

- utilidade;
- coerencia;
- fidelidade a mensagem original;
- adequacao ao tom pedido.

## Usar modelos reais

`configs/models.yaml` contem exemplos para provedores OpenAI-compatible e Ollama. Copie os modelos desejados para `configs/experiment.yaml`.

Para API:

```bash
cp .env.example .env
export OPENAI_API_KEY="..."
```

O manifesto nunca salva chave de API. Ele registra apenas provedor, nome do modelo, parametros, prompts, hash do dataset e ambiente.

## Decisao sobre baseline

O baseline usa prompt direto, mas e repetido `baseline_repetitions` vezes por mensagem. Isso preserva a condicao baseline e permite calcular metricas de diversidade como Jaccard par-a-par e Self-BLEU de forma comparavel ao VS, que gera 5 alternativas.

## Testes

```bash
pytest
```

Os testes cobrem validacao do dataset, parser VS, metricas lexicais e execucao com `FakeLLMClient`.

