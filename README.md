# TP1 NLP - Assistente de Rascunho com Verbalized Sampling

Pipeline experimental em Python para comparar geracao direta (`baseline`) contra Verbalized Sampling (`VS`) em mensagens cotidianas. A entrega oficial usa **dois modelos locais via LM Studio** e **um modelo fechado via OpenAI API**, com 3 condicoes experimentais por modelo.

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

## Modelos

O experimento oficial espera o servidor local de chat do LM Studio ativo para os modelos abertos:

```text
http://localhost:1234/v1
```

Modelos configurados em `configs/experiment.lmstudio.yaml`:

```text
openai_gpt41_mini     -> gpt-4.1-mini
lmstudio_gemma_e2b     -> gemma-4-e2b-it-mlx
lmstudio_ministral_3b  -> ministral-3-3b-instruct-2512
```

Antes de rodar o experimento, abra o LM Studio, carregue Gemma e Ministral e inicie o servidor local. Para o modelo OpenAI, defina `OPENAI_API_KEY` no ambiente ou em `.env`.

O repositório também inclui um fluxo paralelo em inglês em `configs/experiment.lmstudio.en.yaml`. Ele usa os mesmos modelos e parâmetros, mas lê `data/messages_en.jsonl`, usa prompts em inglês e calcula perplexity com `gpt2`.

## Configuracao Experimental

A configuracao oficial usa a comparacao controlada como resultado principal: `temperature` e `top_p` ficam iguais nas tres condicoes para isolar o efeito do prompt de Verbalized Sampling.

```text
baseline         temperature=0.6  top_p=0.95  max_tokens=220
vs_intermediate  temperature=0.6  top_p=0.95  max_tokens=700
vs_aggressive    temperature=0.6  top_p=0.95  max_tokens=700  probability_threshold=0.10
```

O limite de tokens e maior nas condicoes VS porque elas precisam retornar JSON com 5 alternativas. As metricas sao calculadas sobre os textos extraidos de cada alternativa, nao sobre o JSON bruto.

Os prompts VS usam uma lista fechada de tons para facilitar a avaliacao automatica:

```text
formal, cordial, objetivo, direto, empatico, informal, firme, agradecido, positivo, justificativo, curto, gentil, entusiasmado, caloroso, animado
```

No fluxo em ingles, a lista equivalente e:

```text
formal, cordial, objective, direct, empathetic, informal, firm, grateful, positive, justified, short, gentle, enthusiastic, warm
```

## Validacao

Validar estrutura, dataset e prompts:

```bash
python3 scripts/validate_project.py
```

Validar se o LM Studio esta ativo e expondo os 2 modelos locais:

```bash
python3 scripts/check_lmstudio.py --config configs/experiment.lmstudio.yaml
```

Validar o fluxo em ingles:

```bash
python3 scripts/validate_project.py --config configs/experiment.lmstudio.en.yaml
python3 scripts/check_lmstudio.py --config configs/experiment.lmstudio.en.yaml
```

Executar smoke test por modelo:

```bash
python3 scripts/smoke_lmstudio.py --model gemma
python3 scripts/smoke_lmstudio.py --model ministral
```

Cada smoke gera um `run_id` automatico e roda apenas a condicao `baseline` para o modelo local escolhido.

## Execucao Oficial

Rodar o experimento completo:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_pt_final
```

Rodar o experimento completo em ingles:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiment.lmstudio.en.yaml \
  --run-id lmstudio_en_final
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
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_pt_final
```

Para calcular metricas do fluxo em ingles com perplexity:

```bash
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.en.yaml \
  --run-id lmstudio_en_final
```

Gerar graficos e exemplos:

```bash
python3 scripts/make_plots.py --run-id lmstudio_pt_final
python3 scripts/make_plots.py --run-id lmstudio_en_final
```

Saidas principais:

```text
results/runs/lmstudio_pt_final/metrics_long.csv
results/runs/lmstudio_pt_final/metrics_summary.csv
results/runs/lmstudio_pt_final/figures/
results/runs/lmstudio_pt_final/response_examples.csv

results/runs/lmstudio_en_final/metrics_long.csv
results/runs/lmstudio_en_final/metrics_summary.csv
results/runs/lmstudio_en_final/figures/
results/runs/lmstudio_en_final/response_examples.csv
```

Metricas automaticas implementadas:

- `jaccard`: sobreposicao lexical media par-a-par; menor tende a indicar maior diversidade.
- `self_bleu`: repeticao entre respostas; menor tende a indicar maior diversidade.
- `distinct_1` e `distinct_2`: variedade lexical; maior tende a indicar maior diversidade.
- `tone_coverage`: cobertura dos tons esperados por mensagem.
- `parse_success_rate`: taxa de grupos VS com JSON parseado em respostas validas.
- `valid_tone_rate`: proporcao de alternativas VS cujo `tone` pertence a lista fechada.
- `threshold_violation_rate`: proporcao de respostas do VS agressivo com probabilidade maior ou igual ao limite configurado.
- `perplexity`: opcional, com modelo avaliador fixo via `transformers`.

## Avaliacao Manual

Exportar planilha:

```bash
python3 scripts/export_manual_eval.py --run-id lmstudio_pt_final
python3 scripts/export_manual_eval.py --run-id lmstudio_en_final
```

Depois de preencher notas de 1 a 5:

```bash
python3 scripts/import_manual_eval.py \
  --run-id lmstudio_pt_final \
  --file results/runs/lmstudio_pt_final/manual_eval_template.csv
```

Campos avaliados:

- utilidade;
- coerencia;
- fidelidade a mensagem original;
- adequacao ao tom pedido.

## Politica de Resultados

`results/runs/*` e ignorado por padrao para evitar versionar execucoes intermediarias, logs grandes e tentativas quebradas.

Para a entrega, use apenas as duas runs finais:

```text
results/final/lmstudio_pt_final/
results/final/lmstudio_en_final/
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

Nao use runs antigos de debug, smoke tests, runs com Qwen/Phi ou execucoes falhas como resultado oficial.

## Decisao sobre Baseline

O baseline usa prompt direto, mas e repetido `baseline_repetitions` vezes por mensagem. Isso preserva a condicao baseline e permite calcular metricas de diversidade como Jaccard par-a-par e Self-BLEU de forma comparavel ao VS, que gera 5 alternativas.

## Testes

```bash
pytest
```

Os testes automatizados usam `FakeLLMClient` internamente para validar o pipeline sem depender do LM Studio ou da OpenAI API. O experimento oficial usa Gemma e Ministral pelo LM Studio e `gpt-4.1-mini` pela OpenAI API.
