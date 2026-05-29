[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HBpVR1RR)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24014449&assignment_repo_type=AssignmentRepo)

# TP1 NLP - Assistente de Rascunho com Verbalized Sampling

Pipeline experimental em Python para comparar geracao direta (`baseline`) contra Verbalized Sampling (`VS`) em mensagens cotidianas em portugues. A entrega oficial usa dois modelos locais via LM Studio e um modelo fechado via OpenAI API, com tres condicoes experimentais por modelo.

## Estrutura

```text
configs/              Configuracao oficial do experimento
data/                 Dataset em portugues e templates de prompt
scripts/              CLIs de execucao, validacao e analise
src/tp1_nlp/          Codigo modular do pipeline
results/runs/         Artefatos brutos de novas execucoes
results/final/        Run final selecionada para entrega
output/doc/           Relatorio final em DOCX/PDF e figuras usadas no documento
tests/                Testes com pytest
```

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O fluxo oficial calcula perplexity por padrao. Por isso, `requirements.txt` inclui `transformers` e `torch`, usados pelo avaliador fixo de perplexity.

## Ambiente

As runs finais foram geradas em macOS Apple Silicon usando LM Studio como servidor local OpenAI-compatible para os modelos abertos e OpenAI API para o modelo fechado.

Para reproduzir em outro computador, e necessario:

- Python 3.11 ou superior;
- dependencias instaladas via `requirements.txt`;
- LM Studio ativo em `http://localhost:1234/v1`;
- modelos locais configurados disponiveis no LM Studio;
- variavel `OPENAI_API_KEY` configurada para executar `gpt-4.1-mini`.

Se os modelos aparecerem com IDs diferentes no LM Studio, rode o checker e ajuste o campo `model_name` em `configs/experiment.lmstudio.yaml`.

## Modelos Oficiais

Modelos configurados em `configs/experiment.lmstudio.yaml`:

```text
openai_gpt41_mini     -> gpt-4.1-mini
lmstudio_gemma_e2b     -> mlx-community/gemma-4-e2b-it
lmstudio_ministral_14b -> ministral-3-14b-instruct-2512
```

Antes de rodar:

1. Abra o LM Studio.
2. Baixe ou selecione modelos equivalentes a Gemma e Ministral.
3. Inicie o servidor local em `http://localhost:1234/v1`.
4. Rode:

```bash
python3 scripts/check_lmstudio.py --config configs/experiment.lmstudio.yaml
```

Para o modelo OpenAI, defina:

```bash
export OPENAI_API_KEY="sua-chave"
```

Ou crie um `.env` local a partir do exemplo:

```bash
cp .env.example .env
```

O arquivo `.env` e ignorado pelo Git e nao deve ser entregue com chave real.

## Configuracao Experimental

A configuracao oficial usa comparacao controlada: `temperature` e `top_p` ficam iguais nas tres condicoes para isolar o efeito do prompt de VS.

```text
baseline         temperature=0.6  top_p=0.95  max_tokens=220
vs_intermediate  temperature=0.6  top_p=0.95  max_tokens=700
vs_aggressive    temperature=0.6  top_p=0.95  max_tokens=700  probability_threshold=0.10
```

O baseline usa prompt direto, mas e repetido `baseline_repetitions=5` vezes por mensagem. Isso preserva a condicao baseline e permite calcular metricas de diversidade como Jaccard par-a-par e Self-BLEU de forma comparavel ao VS, que gera cinco alternativas.

Os prompts oficiais usam templates `v3`, com mensagens curtas e alvo aproximado de 15 palavras por resposta, inspirado nos prompts do paper de Verbalized Sampling. A mensagem enviada ao modelo contem a mensagem de entrada e a categoria da situacao. Os campos `formality` e `expected_tones` permanecem no dataset como labels ocultos para avaliacao e agrupamento, mas nao sao passados ao modelo.

O VS retorna uma distribuicao verbalizada em JSON com:

```json
{
  "responses": [
    {
      "tone": "cordial",
      "text": "texto da resposta",
      "probability": 0.24
    }
  ]
}
```

O campo `probability` representa a probabilidade estimada da resposta, relativa a distribuicao completa de respostas plausiveis. O campo `tone` e a categoria da situacao sao adaptacoes ao enunciado do TP.

## Validacao

Validar estrutura, dataset e prompts:

```bash
python3 scripts/validate_project.py --config configs/experiment.lmstudio.yaml
```

Validar se o LM Studio esta ativo e expondo os modelos locais:

```bash
python3 scripts/check_lmstudio.py --config configs/experiment.lmstudio.yaml
```

Executar smoke test por modelo local:

```bash
python3 scripts/smoke_lmstudio.py --model gemma
python3 scripts/smoke_lmstudio.py --model ministral
```

## Execucao Oficial

Rodar o experimento completo:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_pt_final
```

Cada execucao cria:

```text
results/runs/<run_id>/
  manifest.json
  run.log
  generations.jsonl
  raw_responses/
```

O `manifest.json` registra dataset, hash do dataset, modelos, parametros, prompts, ambiente e caminhos dos arquivos gerados.

## Metricas e Graficos

Calcular metricas automaticas completas, incluindo perplexity:

```bash
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_pt_final
```

Para desenvolvimento rapido, e possivel pular perplexity:

```bash
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id lmstudio_pt_final \
  --skip-perplexity
```

Gerar graficos e exemplos:

```bash
python3 scripts/make_plots.py --run-id lmstudio_pt_final
```

Saidas principais:

```text
results/runs/lmstudio_pt_final/metrics_long.csv
results/runs/lmstudio_pt_final/metrics_summary.csv
results/runs/lmstudio_pt_final/figures/
results/runs/lmstudio_pt_final/response_examples.csv
```

Metricas automaticas implementadas:

- `jaccard`: sobreposicao lexical media par-a-par; menor tende a indicar maior diversidade.
- `self_bleu`: repeticao entre respostas; menor tende a indicar maior diversidade.
- `distinct_1` e `distinct_2`: variedade lexical; maior tende a indicar maior diversidade.
- `tone_diversity_ratio`: tons validos distintos divididos pelo numero de respostas validas.
- `tone_coverage`: cobertura dos tons esperados ocultos por mensagem; metrica auxiliar.
- `parse_success_rate`: taxa de grupos VS com JSON parseado em respostas validas.
- `valid_tone_rate`: proporcao de alternativas VS cujo `tone` pertence a lista fechada.
- `threshold_violation_rate`: proporcao de respostas do VS agressivo com probabilidade maior ou igual ao limite configurado.
- `perplexity`: fluencia estimada com `PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit`.

## Avaliacao Manual

Exportar planilha:

```bash
python3 scripts/export_manual_eval.py --run-id lmstudio_pt_final
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
- adequacao ao tom declarado.

Escala usada na avaliacao manual:

```text
5 = excelente, pronta para envio.
4 = boa, com pequeno ajuste possivel.
3 = aceitavel, mas precisa edicao.
2 = fraca, com problema relevante.
1 = inadequada.
```

Interpretacao dos campos:

- `utility`: mede se a resposta seria util em uma conversa real. Pequenas inferencias plausiveis sao permitidas.
- `coherence`: mede se o texto e claro, natural e bem formado.
- `fidelity`: mede se a resposta respeita a intencao central da mensagem. Inferencias leves sao aceitaveis, mas mudancas de intencao devem ser penalizadas.
- `tone_adequacy`: mede se a resposta combina com a formalidade e os tons esperados.

Diversidade nao foi tratada como melhoria automatica: uma resposta diferente so e boa se continuar util, coerente e adequada.

## Resultado Final Entregue

A run oficial selecionada para entrega esta em:

```text
results/final/resultados_finais/
```

Artefatos incluidos:

```text
manifest.json
generations.jsonl
metrics_long.csv
metrics_summary.csv
figures/
response_examples.csv
manual_eval_template.csv
manual_eval_completed.csv
manual_metrics_summary.csv
manual_metrics_by_category.csv
observações.md
```

O relatorio final foi gerado fora de `results/`, em:

```text
output/doc/
  relatorio_tp1_nlp.docx
  relatorio_tp1_nlp.pdf
  assets/
```

`relatorio_tp1_nlp.docx` e o documento principal para edicao e entrega. `relatorio_tp1_nlp.pdf` e a versao exportada pelo LibreOffice para conferencia visual. A pasta `assets/` contem as figuras incorporadas ao relatorio, incluindo o fluxo do pipeline, comparacoes de metricas automaticas, avaliacao manual por modelo/condicao, ganho por categoria e comparacao por formalidade.

`results/runs/*` e ignorado por padrao para evitar versionar execucoes intermediarias, logs grandes e tentativas descartadas. Para reproduzir do zero, execute os comandos oficiais e gere uma nova pasta em `results/runs/`.

## Testes

```bash
pytest
```

Os testes automatizados usam `FakeLLMClient` internamente para validar o pipeline sem depender do LM Studio ou da OpenAI API.
