# Observações críticas da avaliação manual

## Base de avaliação

A avaliação manual seguiu uma rubrica para mensagens curtas e cotidianas, separando utilidade prática de fidelidade estrita ao contexto. Foram avaliadas 1076 saídas da run `resultados_finais`, incluindo uma falha de parse como resposta não pronta para uso.

A especificação do TP pede comparação entre baseline e Verbalized Sampling (VS) quanto a diversidade lexical, diversidade de tom, repetição, adequação ao contexto e utilidade prática. Também pede discussão sobre diferenças entre mensagens formais e informais, variação real versus paráfrase superficial e situações em que o VS ajuda mais.

## Resultado geral

O VS aumentou claramente a diversidade, mas isso veio com perda moderada de fidelidade e utilidade em alguns grupos.

Médias automáticas por condição:

| Condição | Jaccard ↓ | Self-BLEU ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Diversidade de tom ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.480 | 0.649 | 0.422 | 0.539 | 0.000 |
| vs_intermediate | 0.136 | 0.197 | 0.712 | 0.902 | 0.981 |
| vs_aggressive | 0.102 | 0.150 | 0.757 | 0.943 | 0.997 |

Médias manuais por condição:

| Condição | Utilidade | Coerência | Fidelidade | Tom |
| --- | ---: | ---: | ---: | ---: |
| baseline | 4.458 | 4.928 | 4.208 | 4.950 |
| vs_intermediate | 4.160 | 4.914 | 3.956 | 4.946 |
| vs_aggressive | 4.017 | 4.939 | 3.828 | 4.925 |

Interpretação: o VS cumpriu o objetivo de reduzir repetição e ampliar alternativas, mas o baseline continuou mais confiável em utilidade e fidelidade. A configuração agressiva foi a mais diversa, porém também a que mais assumiu detalhes não dados.

## Diferenças entre modelos

Médias manuais gerais:

| Modelo | Utilidade | Coerência | Fidelidade | Tom |
| --- | ---: | ---: | ---: | ---: |
| openai_gpt41_mini | 4.361 | 5.000 | 4.103 | 4.983 |
| lmstudio_gemma_e2b | 4.267 | 5.000 | 4.011 | 4.933 |
| lmstudio_ministral_14b | 4.007 | 4.780 | 3.878 | 4.904 |

O `openai_gpt41_mini` foi o mais estável em coerência e adequação ao tom. O `gemma_e2b` gerou saídas limpas e coerentes, mas o VS dele falhou mais em recusa, frequentemente trocando a recusa por "vou verificar". O `ministral_14b` trouxe diversidade forte, mas teve mais problemas de markdown, placeholders, datas inventadas e uma falha de parse.

## Onde o VS ajudou

O VS ajudou mais em mensagens com alto potencial de variação: cobrança, convite, adiamento e alguns pedidos. Nesses casos, as alternativas trouxeram estratégias úteis como pedir prazo, suavizar cobrança, propor nova data, responder de forma mais empática ou mais direta.

O ganho é especialmente visível em Distinct-2 e Self-BLEU: as respostas VS são muito menos parecidas entre si do que as cinco repetições do baseline. Isso atende bem ao foco do trabalho: reduzir respostas seguras e repetitivas.

## Onde o VS prejudicou

O VS prejudicou quando a categoria exigia uma intenção muito específica. Em recusa, várias respostas de VS não recusaram: apenas disseram que iriam checar agenda ou confirmar depois. Isso aumenta diversidade superficialmente, mas perde a função comunicativa principal.

Também houve queda em esclarecimento, especialmente na mensagem sobre prazo entre sexta e segunda. Muitas respostas assumiram uma data como correta. Pela rubrica usada, isso pode continuar útil em um contexto plausível, mas perde fidelidade por assumir informação relevante não dada.

No VS agressivo, apareceram detalhes específicos demais: datas como `21/06`, `25/06`, `15/07`, motivos pessoais, "coordenação", "edital", "sistema fecha", restaurante/garçom/cardápio e marcas de ação como `*Suspiro de alívio*`. Esses casos aumentam variedade, mas nem sempre são boas alternativas para envio.

## Problemas recorrentes

Contagem de observações nas notas manuais:

| Problema | Ocorrências |
| --- | ---: |
| assume detalhe relevante não dado | 83 |
| não agradece explicitamente | 79 |
| marcas de markdown ou meta-texto | 33 |
| assume resposta para informação não explicitada | 25 |
| adiou a decisão em vez de recusar | 8 |
| estilo excessivo/deslocado | 7 |
| não recusa claramente | 5 |
| placeholder não resolvido | 4 |
| responde como confirmação, não adiamento | 3 |
| falha de parse | 1 |

O problema de agradecimento é importante: em várias mensagens da categoria, o modelo respondeu como se ele tivesse ajudado ("Fico feliz em ajudar") em vez de agradecer a ação do interlocutor. Isso reduz fidelidade, porque muda o papel comunicativo da resposta.

## Formalidade e tom

A adequação ao tom foi alta em todas as condições, mas a média esconde casos ruins. As mensagens formais sofreram mais quando o VS agressivo gerou informalidade excessiva, emoji, markdown ou expressões como "meu amigo" e "Whats". Em mensagens informais, essa liberdade é menos problemática, mas ainda pode soar deslocada quando o conteúdo inventa contexto demais.

O baseline tende a respeitar melhor a formalidade porque fica mais perto da resposta padrão. O VS amplia opções de tom, mas exige filtragem posterior para remover alternativas teatralizadas ou íntimas demais.

## Conclusão crítica

Os resultados sustentam a hipótese central do TP: o VS aumenta diversidade lexical e de tom de forma clara. Porém, diversidade não é melhoria automática. A configuração intermediária parece o melhor compromisso: melhora muito a diversidade em relação ao baseline e preserva mais utilidade/fidelidade do que o VS agressivo.

Para uso real como assistente de rascunho, o VS seria mais valioso como gerador de opções com pós-filtro: descartar respostas que não cumprem a intenção da categoria, que inventam datas/motivos, que usam markdown ou que mudam formalidade. Sem esse filtro, o baseline ainda é mais seguro em mensagens simples e em categorias com intenção rígida, como recusa, confirmação e agradecimento.
