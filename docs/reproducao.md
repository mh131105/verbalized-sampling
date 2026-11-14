# Reprodução do experimento

[Visão geral e resultados](../README.md) · [Relatório final](../output/doc/relatorio_tp1_nlp.pdf)

Execute os comandos a partir da raiz do repositório. A geração usa modelos reais e pode consumir créditos da API. Os IDs registrados não fixam todas as revisões e quantizações dos modelos locais; alterar esses componentes muda as condições de reprodução.


Pipeline experimental em Python para comparar geração direta (`baseline`) contra Verbalized Sampling (`VS`) em mensagens cotidianas em português. A entrega oficial usa dois modelos locais via LM Studio e um modelo fechado via OpenAI API, com três condições experimentais por modelo.

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

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O fluxo oficial calcula perplexity por padrão. Por isso, `requirements.txt` inclui `transformers` e `torch`, usados pelo avaliador fixo de perplexity.

## Ambiente

As runs finais foram geradas em macOS Apple Silicon usando LM Studio como servidor local OpenAI-compatible para os modelos abertos e OpenAI API para o modelo fechado.

Para reproduzir em outro computador, e necessario:

- Python 3.11 ou superior;
- dependencias instaladas via `requirements.txt`;
- LM Studio ativo em `http://localhost:1234/v1`;
- modelos locais configurados disponiveis no LM Studio;
- variável `OPENAI_API_KEY` configurada para executar `gpt-4.1-mini`.

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
2. Baixe ou selecione os modelos registrados na configuração. Alterar o modelo ou a quantização modifica o experimento.
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

O arquivo `.env` e ignorado pelo Git e não deve ser entregue com chave real.

## Configuração Experimental

Temperatura e top-p permanecem iguais nas três condições. A comparação também varia instruções, formato da resposta, número de chamadas e limite de saída.

```text
baseline         temperature=0.6  top_p=0.95  max_tokens=220
vs_intermediate  temperature=0.6  top_p=0.95  max_tokens=700
vs_aggressive    temperature=0.6  top_p=0.95  max_tokens=700  probability_threshold=0.10
```

O baseline usa prompt direto, mas e repetido `baseline_repetitions=5` vezes por mensagem. Isso preserva a condição baseline e permite calcular métricas de diversidade como Jaccard par-a-par e Self-BLEU de forma comparavel ao VS, que gera cinco alternativas.

Os prompts oficiais usam templates `v3`, com mensagens curtas e alvo aproximado de 15 palavras por resposta, inspirado nos prompts do paper de Verbalized Sampling. A mensagem enviada ao modelo contem a mensagem de entrada e a categoria da situacao. Os campos `formality` e `expected_tones` permanecem no dataset como labels ocultos para avaliação e agrupamento, mas não são passados ao modelo.

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

O campo `probability` representa a probabilidade estimada da resposta, relativa a distribuicao completa de respostas plausiveis. O campo `tone` e a categoria da situacao são adaptacoes ao enunciado do TP.

## Validação

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

## Execução Oficial

Rodar o experimento completo:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id reproducao_vs
```

Cada execução cria:

```text
results/runs/<run_id>/
  manifest.json
  run.log
  generations.jsonl
  raw_responses/
```

O `manifest.json` registra dataset, hash do dataset, modelos, parâmetros, prompts, ambiente e caminhos dos arquivos gerados.

## Métricas e Gráficos

Calcular métricas automaticas completas, incluindo perplexity:

```bash
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id reproducao_vs
```

Para desenvolvimento rápido, e possível pular perplexity:

```bash
.venv/bin/python scripts/compute_metrics.py \
  --config configs/experiment.lmstudio.yaml \
  --run-id reproducao_vs \
  --skip-perplexity
```

Gerar gráficos e exemplos:

```bash
python3 scripts/make_plots.py --run-id reproducao_vs
```

Saidas principais:

```text
results/runs/reproducao_vs/metrics_long.csv
results/runs/reproducao_vs/metrics_summary.csv
results/runs/reproducao_vs/figures/
results/runs/reproducao_vs/response_examples.csv
```

Métricas automaticas implementadas:

- `jaccard`: sobreposição lexical media par-a-par; menor tende a indicar maior diversidade.
- `self_bleu`: repetição entre respostas; menor tende a indicar maior diversidade.
- `distinct_1` e `distinct_2`: variedade lexical; maior tende a indicar maior diversidade.
- `tone_diversity_ratio`: tons validos distintos divididos pelo número de respostas validas.
- `tone_coverage`: cobertura dos tons esperados ocultos por mensagem; metrica auxiliar.
- `parse_success_rate`: taxa de grupos VS com JSON parseado em respostas validas.
- `valid_tone_rate`: proporção de alternativas VS cujo `tone` pertence a lista fechada.
- `threshold_violation_rate`: proporção de respostas do VS agressivo com probabilidade maior ou igual ao limite configurado.
- `perplexity`: fluência estimada com `PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit`.

## Avaliação Manual

Exportar planilha:

```bash
python3 scripts/export_manual_eval.py --run-id reproducao_vs
```

Depois de preencher notas de 1 a 5:

```bash
python3 scripts/import_manual_eval.py \
  --run-id reproducao_vs \
  --file results/runs/reproducao_vs/manual_eval_template.csv
```

Campos avaliados:

- utilidade;
- coerencia;
- fidelidade a mensagem original;
- adequação ao tom declarado.

Escala usada na avaliação manual:

```text
5 = excelente, pronta para envio.
4 = boa, com pequeno ajuste possivel.
3 = aceitavel, mas precisa edicao.
2 = fraca, com problema relevante.
1 = inadequada.
```

Interpretacao dos campos:

- `utility`: mede se a resposta seria útil em uma conversa real. Pequenas inferências plausiveis são permitidas.
- `coherence`: mede se o texto e claro, natural e bem formado.
- `fidelity`: mede se a resposta respeita a intenção central da mensagem. Inferências leves são aceitaveis, mas mudancas de intenção devem ser penalizadas.
- `tone_adequacy`: mede se a resposta combina com a formalidade e os tons esperados.

Diversidade não foi tratada como melhoria automatica: uma resposta diferente so e boa se continuar útil, coerente e adequada.

## Resultado Final Entregue

A run oficial selecionada para entrega esta em:

```text
results/final/resultados_finais/
```

Artefatos incluídos:

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

O relatório final foi gerado fora de `results/`, em:

```text
output/doc/
  relatorio_tp1_nlp.docx
  relatorio_tp1_nlp.pdf
  assets/
```

`relatorio_tp1_nlp.docx` e o documento principal para edição e entrega. `relatorio_tp1_nlp.pdf` e a versão exportada pelo LibreOffice para conferência visual. A pasta `assets/` contem as figuras incorporadas ao relatório, incluindo o fluxo do pipeline, comparacoes de métricas automaticas, avaliação manual por modelo/condição, ganho por categoria e comparação por formalidade.

`results/runs/*` e ignorado por padrão para evitar versionar execuções intermediárias, logs grandes e tentativas descartadas. Para reproduzir do zero, execute os comandos oficiais e gere uma nova pasta em `results/runs/`.

## Testes

```bash
pytest
```

Os testes automatizados usam `FakeLLMClient` internamente para validar o pipeline sem depender do LM Studio ou da OpenAI API.

## Cuidados ao interpretar os resultados

A execução final possui 1.076 registros: 1.075 com status `success` e uma falha de parse. A falha recebeu nota 1 na avaliação registrada e substituiu cinco alternativas esperadas por um registro. As métricas de diversidade do Ministral/VS intermediário usam 23 grupos; as notas desse grupo abrangem 116 registros. As outras combinações usam 24 grupos e 120 registros manuais.

O baseline não declara `tone`. O zero em `tone_diversity_ratio` não demonstra ausência de diversidade real de tom e não permite comparação direta com VS. As probabilidades verbalizadas são estimativas declaradas, sem calibração demonstrada. Métricas lexicais e perplexidade não substituem a leitura de utilidade e fidelidade.

As notas originais não são atribuídas às respostas de uma nova execução. O relatório registra um avaliador; consulte as limitações na [visão geral](../README.md#limitações).
