// Compilar a partir da raiz do repositório (o artigo importa arquivos de results/):
//   typst watch --root . artigo/main.typ
//
// Modelo IEEE de conferência (ieee.typ), indicado pela professora.

#import "ieee.typ": ieee
#import "../results/summary/tables.typ": *

#show: ieee.with(
  title: [Verificação de Locutor com ResNet-34 e _Fine-Tuning_],
  authors: (
    (name: "Raul Rodrigues", organization: [Universidade Tecnológica Federal do Paraná], location: [Ponta Grossa, Brasil]),
    (name: "Paulo Henrique", organization: [Universidade Tecnológica Federal do Paraná], location: [Ponta Grossa, Brasil]),
  ),
  abstract: [Este trabalho avalia uma ResNet-34 pré-treinada na VoxCeleb2 para verificação de locutor 1:1, antes e depois de um _fine-tuning_ com perda AAM-Softmax na LibriSpeech. As representações vetoriais (_embeddings_) são comparadas por similaridade de cosseno, com limiar de decisão (_threshold_) congelado na validação e protocolo com locutores disjuntos. O ajuste reduziu o EER de teste na LibriSpeech de 0,395% para 0,201 $plus.minus$ 0,017% (média de três _seeds_) e manteve EER menor na VCTK e sob ruído branco de 20, 10 e 0 dB. Com o limiar congelado, porém, a FMR do modelo ajustado foi maior fora da condição de treinamento, chegando a 3,938 $plus.minus$ 0,435% na VCTK, contra 1,825% do modelo pré-treinado. Os resultados indicam que um EER menor não garante um ponto de operação robusto e que o limiar deve ser calibrado para a condição de uso.],
  index-terms: ("biometria", "verificação de locutor", [_embeddings_], [_fine-tuning_], "redes neurais convolucionais"),
  bibliography: bibliography("refs.bib"),
  paper-size: "a4",
)
#set text(lang: "pt", region: "br")
#set figure(placement: auto)

= Introdução

A verificação de locutor é uma tarefa biométrica que consiste em decidir, a partir de duas gravações de voz, se elas pertencem à mesma pessoa. Trata-se de uma comparação 1:1 e não deve ser confundida com o reconhecimento de fala: o objetivo não é descobrir o que foi dito, mas quem falou. A voz é uma modalidade atraente, pois pode ser capturada sem contato, com microfones comuns e à distância, como em atendimentos por telefone e em assistentes virtuais. Em contrapartida, o sinal é afetado pelo microfone, pelo ambiente, pelo ruído e pela própria variação vocal do falante.

Os sistemas atuais utilizam redes neurais profundas para mapear cada gravação em uma representação vetorial de dimensão fixa, denominada _embedding_, que é comparada por uma medida de similaridade. Entretanto, o desempenho desses modelos costuma ser resumido pelo EER, uma métrica que não depende de um limiar de decisão (_threshold_) previamente fixado. Um sistema real, por outro lado, opera com um limiar definido antes do uso, e um bom EER não garante que esse valor continue adequado quando as condições mudam.

O objetivo deste trabalho é avaliar uma ResNet-34 pré-treinada na VoxCeleb2, antes e depois de um _fine-tuning_ na LibriSpeech, para verificação de locutor com similaridade de cosseno. As principais contribuições são:

- um protocolo reprodutível, com locutores disjuntos entre os conjuntos, para duas bases que não possuem protocolo oficial de verificação;
- uma comparação entre o modelo pré-treinado e o ajustado, com três _seeds_, em condições limpas, entre bases e com ruído;
- uma análise do ponto de operação, que mostra que a redução do EER obtida com o _fine-tuning_ pode vir acompanhada de um aumento da FMR fora da condição de treinamento;
- uma análise de erros que identifica um provável erro de rótulo na LibriSpeech e casos de variação na voz de um mesmo locutor.

= Trabalhos relacionados

Durante anos, a verificação de locutor foi dominada pelos _i-vectors_ @dehak2011ivector, que representam cada gravação em um espaço de baixa dimensão que modela conjuntamente a variabilidade de locutor e de canal. Com o avanço das redes neurais, essas representações passaram a ser aprendidas diretamente dos dados. Os _d-vectors_ @variani2014dvector utilizam a última camada oculta de uma rede treinada para classificar locutores, e os _x-vectors_ @snyder2018xvector mapeiam áudios de duração variável para _embeddings_ de dimensão fixa por meio de uma camada de agregação estatística (_pooling_).

As CNNs residuais @he2016resnet tornaram-se uma das principais arquiteturas nessa tarefa, por tratarem o espectrograma como uma entrada bidimensional. No desafio VoxSRC 2019, o sistema da Universidade de Tecnologia de Brno (BUT) @zeinali2019but combinou redes ResNet-34 com redes baseadas em _x-vectors_ e obteve EER de 1,42% na condição fixa e de 1,26% na condição aberta. Posteriormente, a ECAPA-TDNN @desplanques2020ecapa incorporou atenção por canal e agregação de características de múltiplas camadas às redes de atraso temporal (TDNN). Paralelamente, perdas com margem angular, como a ArcFace @deng2019arcface, proposta para reconhecimento facial, foram adotadas para produzir _embeddings_ mais discriminativos, que podem ser comparados diretamente por similaridade de cosseno.

Na VoxCeleb1, a ECAPA-TDNN alcançou EER de 0,87%, contra 1,19% de uma ResNet-34 avaliada no mesmo trabalho @desplanques2020ecapa. Em um relatório mais recente do WeSpeaker @wang2023baselines, a ResNet-34 atingiu 0,723% na lista VoxCeleb1-O. Esses valores servem apenas como referência, pois dependem da base, do protocolo e do pós-processamento de escores (_scores_), e não são diretamente comparáveis aos deste trabalho.


= Base de dados

Este trabalho utiliza duas bases públicas de fala em inglês, ambas distribuídas sob a licença CC BY 4.0. A LibriSpeech é a base principal, usada no ajuste e na avaliação do modelo, e a VCTK é usada apenas na avaliação entre bases (_cross-dataset_). Como nenhuma delas possui um protocolo oficial de verificação de locutor, foi construído um protocolo próprio, descrito na @sec:protocolo. A @tab:bases resume suas características, e a @fig:pipeline mostra as etapas do pré-processamento em uma amostra real.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header([*Característica*], [*LibriSpeech*], [*VCTK*]),
    [Versão/subconjunto], [`train-clean-100`], [0.92, `mic1`],
    [Locutores], [251 (125 F, 126 M)], [110 (63 F, 47 M)],
    [Amostras], [28.539], [44.455],
    [Taxa de amostragem], [16 kHz], [48 kHz],
    [Duração mediana], [14,0 s], [3,1 s],
    [Faixa de duração], [1,4 a 24,5 s], [1,2 a 16,6 s],
    [Aquisição], [Audiolivros], [Estúdio],
  ),
  caption: [Características das bases utilizadas. F: feminino; M: masculino. Durações aproximadas.],
) <tab:bases>

== LibriSpeech

A LibriSpeech @panayotov2015librispeech é uma base de dados construída a partir de audiolivros de domínio público do projeto LibriVox, lidos por voluntários. Foi utilizado o subconjunto `train-clean-100`, com cerca de 100 horas de fala.

Por se originar de gravações feitas pelos próprios leitores, a base apresenta variação de microfone e de ambiente entre locutores. As amostras de um locutor são organizadas por capítulo, o que permite distinguir comparações em um mesmo capítulo e entre capítulos diferentes.

== VCTK

A VCTK @yamagishi2019vctk reúne gravações em estúdio, a 48 kHz, de 110 falantes de inglês com sotaques variados. No protocolo deste trabalho, foram utilizadas somente as gravações do `mic1`, pois os locutores p280 e p315 não possuem o `mic2`.

Os áudios da versão 0.92 já são distribuídos com o silêncio das bordas removido. Além disso, a duração mediana das falas da VCTK é inferior a um quarto da observada na LibriSpeech. Essas diferenças de duração, de sotaque e de condição de gravação tornam a VCTK adequada para avaliar a capacidade de generalização do modelo.

#figure(
  image("../results/pipeline.png", width: 72%),
  scope: "parent",
  caption: [Etapas do pré-processamento para a amostra 7190-90543-0043 da LibriSpeech.]
) <fig:pipeline>

= Metodologia

== Pré-processamento

A detecção de atividade de voz (VAD) é baseada em energia e opera em quadros de 25 ms com deslocamento de 10 ms. Quadros com energia mais de 40 dB inferior à do quadro mais forte do áudio são considerados silêncio, e cada região de fala é estendida em 100 ms para cada lado. Um áudio com menos de 0,5 s de fala após o VAD é contabilizado como falha de aquisição (_Failure to Acquire_, FTA).

Do sinal resultante são extraídas 80 bandas log-Mel no padrão Kaldi, com janela de Hamming de 25 ms, deslocamento de 10 ms e sem _dither_. Por fim, a média de cada banda ao longo do áudio é subtraída. A entrada da rede é, portanto, uma matriz de dimensão $T times 80$, em que $T$ é o número de quadros.

== Arquitetura da CNN

Foi utilizada uma ResNet-34 @he2016resnet na variante estreita do WeSpeaker @wang2023wespeaker, adaptada para espectrogramas. Após uma convolução inicial de 32 canais, a rede possui quatro estágios com 3, 4, 6 e 3 blocos residuais e 32, 64, 128 e 256 canais, respectivamente. Uma agregação estatística concatena a média e o desvio padrão das ativações ao longo do tempo, o que permite processar áudios de duração variável, e uma camada linear produz o _embedding_ de 256 dimensões.

Os pesos iniciais são os do modelo público do WeSpeaker, treinado com ArcFace no conjunto de desenvolvimento da VoxCeleb2 @chung2018voxceleb2, que contém 5.994 locutores. O repositório dos pesos é marcado com a licença Apache-2.0, e o WeSpeaker informa que seus modelos treinados na VoxCeleb seguem a licença CC BY 4.0. Esse modelo, sem nenhum ajuste, também é avaliado como referência.

== _Fine-tuning_

O ajuste foi realizado com os 175 locutores de treino da LibriSpeech, que somam 19.720 amostras, utilizando a perda AAM-Softmax (ArcFace) @deng2019arcface:

$
  cal(L) = -log frac(e^(s cos(theta_y + m)), e^(s cos(theta_y + m)) + sum_(j != y) e^(s cos theta_j)),
$<eq:aam>

em que $theta_j$ é o ângulo entre o _embedding_ e o vetor da classe $j$, $y$ é a classe correta, $m = "0,2"$ é a margem angular e $s = 32$ é o fator de escala. A camada de classificação é usada apenas no treinamento e descartada em seguida.

O treinamento durou 15 épocas, com lotes de 128 recortes aleatórios de 2 s extraídos após o VAD. Na primeira época, somente a camada de classificação foi treinada, com o restante da rede congelado. Utilizou-se SGD com momento de Nesterov de 0,9, decaimento de pesos de $10^(-4)$ e redução cossenoidal da taxa de aprendizado, cujo valor inicial foi de $10^(-3)$ para a rede e de $10^(-2)$ para a camada de classificação.

Ao fim de cada época, o modelo foi avaliado no conjunto de validação. Foi selecionada a época com menor EER, usando a maior AUC como critério de desempate.

== _Embedding_ e comparação

O _embedding_ de cada amostra é normalizado pela norma L2, e o escore $S$ entre duas amostras é a similaridade de cosseno, equivalente ao produto interno dos vetores normalizados. O par é aceito como genuíno se $S >= tau$, em que $tau$ é o limiar congelado na validação (@sec:protocolo).

= Protocolo experimental <sec:protocolo>

== Divisões e pares

Os locutores de cada base foram divididos em treino, validação e teste na proporção 70/15/15, com estratificação por gênero e sem locutores em comum entre os conjuntos. A divisão resultou em 175, 38 e 38 locutores na LibriSpeech e em 78, 16 e 16 na VCTK.

Na validação e no teste, foram sorteados 100 pares genuínos por locutor e 10 pares impostores para cada par genuíno, sempre entre locutores do mesmo conjunto. Os impostores podem ser de qualquer gênero, e os pares genuínos da LibriSpeech podem vir do mesmo capítulo ou de capítulos diferentes. Cada conjunto avaliado contém 3.800 pares genuínos e 38.000 impostores na LibriSpeech, e 1.600 genuínos e 16.000 impostores na VCTK.

== _Seeds_ e hardware

O protocolo foi gerado com a _seed_ 42, e o _fine-tuning_ foi repetido com as _seeds_ 42, 123 e 2026. Seus resultados são reportados como média e desvio padrão amostral das três execuções.

O treinamento e as avaliações foram executados no Kaggle, com GPU Tesla T4, processador Intel Xeon de 2,00 GHz com 4 vCPUs, Python 3.13, PyTorch 2.11.0 e CUDA 13.0. A única exceção é a validação do modelo pré-treinado, executada em um computador local com AMD Ryzen 9 3900X e GPU AMD Radeon RX 6700 XT.

== Limiar de decisão

O limiar de cada modelo é o valor de escore correspondente ao EER na validação da LibriSpeech. Ele é salvo em arquivo e congelado antes da avaliação de qualquer conjunto de teste. Assim, o teste não participa da escolha de limiar, de hiperparâmetros ou de época.

== Avaliação entre bases

Na avaliação entre bases, os modelos são aplicados aos pares de teste da VCTK sem nenhum ajuste adicional, mantendo o limiar da validação da LibriSpeech.

== Pares da outra dupla

Para comparar as duas CNNs nos mesmos dados, os modelos também foram avaliados nos pares de teste da VCTK construídos pela Dupla 7 da disciplina, que utiliza a VCTK como base principal. Esses pares envolvem 17 locutores, com 170 pares genuínos e 5.000 impostores, e incluem os dois microfones da base: 71 genuínos e 2.500 impostores comparam uma gravação do `mic1` com uma do `mic2`. Os pares foram importados sem alteração, em um _commit_ fixo do repositório da outra dupla, e avaliados com o mesmo limiar congelado da validação da LibriSpeech.

== Robustez e ablação

A robustez foi avaliada nos pares de teste da LibriSpeech com ruído branco gaussiano aditivo, em relações sinal-ruído (SNR) de 20, 10 e 0 dB, calculadas sobre a potência de cada áudio. O ruído é adicionado à forma de onda antes do VAD, com uma _seed_ derivada do nome do arquivo, para que todos os modelos recebam o mesmo sinal degradado. Mantém-se o limiar obtido na validação sem ruído.

Como estudo de ablação, o _pipeline_ foi executado sem o VAD. Por se tratar de uma alteração no próprio _pipeline_, essa configuração utiliza um limiar próprio, obtido na validação da LibriSpeech também sem VAD.

= Resultados

== Desempenho de verificação

A @tab:padrao apresenta os resultados no teste da LibriSpeech. O modelo pré-treinado obteve EER de 0,395%, e o _fine-tuning_ reduziu esse valor para 0,201 $plus.minus$ 0,017%. No limiar congelado, a FMR passou de 1,255% para 0,660 $plus.minus$ 0,031%, e a FNMR, de 0,237% para 0,079 $plus.minus$ 0,053%. Não houve falhas de aquisição em nenhuma condição avaliada.

Na validação, o EER foi de 1,211% para o modelo pré-treinado e de 1,158 $plus.minus$ 0,000% para o ajustado. As Figs. @fig:roc[] e @fig:dist[] mostram a curva ROC e a distribuição de escores.

#figure(
  tabela_padronizada,
  scope: "parent",
  caption: [Resultados padronizados no teste da LibriSpeech e da VCTK (avaliação entre bases).],
) <tab:padrao>

#figure(
  image("../results/finetuned_seed42/librispeech_test/roc.png"),
  caption: [Curva ROC no teste da LibriSpeech (modelo ajustado, _seed_ 42), com eixo vertical a partir de 0,98.],
) <fig:roc>

#figure(
  image("../results/finetuned_seed42/librispeech_test/distributions.png"),
  caption: [Distribuição de escores no teste da LibriSpeech (modelo ajustado, _seed_ 42), com o limiar marcado.],
) <fig:dist>

== Avaliação entre bases

Na VCTK (@tab:padrao), o EER foi de 0,687% para o modelo pré-treinado e de 0,581 $plus.minus$ 0,038% para o ajustado. Com o limiar congelado, porém, os modelos se comportaram de forma diferente: a FMR do modelo ajustado foi de 3,938 $plus.minus$ 0,435%, contra 1,825% do pré-treinado, enquanto sua FNMR foi de 0,042 $plus.minus$ 0,036%, contra 0,312%.

Nos pares da Dupla 7 (@tab:resultados), o EER foi de 0,724% para o modelo pré-treinado e de 0,611 $plus.minus$ 0,046% para o ajustado. O padrão se repetiu: a FMR do modelo ajustado foi de 4,393 $plus.minus$ 0,530%, contra 3,020% do pré-treinado, e nenhum dos modelos rejeitou pares genuínos no limiar congelado.

== Robustez e ablação

A @tab:resultados e a @fig:robustez mostram o efeito do ruído. O EER aumentou com a degradação nos dois modelos e, em todos os níveis, foi menor no modelo ajustado.

No limiar congelado, a FMR do modelo ajustado cresceu mais rapidamente, chegando a 52,820 $plus.minus$ 23,859% a 0 dB, contra 10,808% do pré-treinado. O desvio padrão elevado indica que esse efeito varia bastante entre as _seeds_.

Na ablação, a remoção do VAD não alterou o EER do modelo pré-treinado (0,395%). No modelo ajustado, o EER foi de 0,187 $plus.minus$ 0,024%, contra 0,201 $plus.minus$ 0,017% com VAD, uma diferença menor que o desvio padrão entre as execuções.

#figure(
  tabela_resultados,
  scope: "parent",
  caption: [Resultados por condição. Modelo ajustado: média ± desvio padrão de 3 _seeds_.],
) <tab:resultados>

#figure(
  image("../results/summary/robustness.png", width: 90%),
  scope: "parent",
  caption: [EER e FMR sob ruído branco aditivo, com o limiar congelado da validação.],
) <fig:robustez>

== Eficiência

O modelo possui 6.634.336 parâmetros e ocupa 25,4 MB. No Kaggle, o modelo ajustado levou em média, por amostra da LibriSpeech, 24,6 $plus.minus$ 0,7 ms no pré-processamento em CPU, incluindo a leitura do arquivo, e 31,6 $plus.minus$ 0,0 ms na extração do _embedding_ na GPU Tesla T4.

= Análise e discussão

== Análise de erros

Os erros foram analisados no modelo ajustado com a _seed_ 42, escolhida antes da inspeção dos casos. No teste da LibriSpeech, com limiar de 0,314, ocorreram 237 falsas correspondências (_false matches_) e 5 falsas não correspondências (_false non-matches_). Das falsas correspondências, 216 envolvem locutores do mesmo gênero, o que é compatível com uma confusão causada por vozes semelhantes. A @fig:erros apresenta as três falsas correspondências de maior escore e as três falsas não correspondências de menor escore.

A falsa correspondência de maior escore (0,569) compara as amostras 6209-34601-0010 e 8747-293952-0060. Na inspeção auditiva, a primeira aparenta conter o leitor interpretando a voz de um personagem, mas não é possível afirmar que isso tenha causado o erro. Os outros dois casos comparam os locutores 2893 e 6415 (0,483), de gêneros diferentes, e 3486 e 8580 (0,453). Como todos os áudios têm ao menos 13 s e mais de 84% de fala, a duração é uma causa pouco provável. A hipótese é de semelhança de timbre ou de condição de gravação, que não foi confirmada.

As cinco falsas não correspondências pertencem ao locutor 6209, e os três casos da figura (0,282, 0,292 e 0,297) comparam os capítulos 34599 e 34601. O capítulo 34601 é pouco consistente até consigo mesmo: seu escore genuíno médio interno é de 0,563, contra 0,871 e 0,874 nos outros dois capítulos do locutor. Na amostra 6209-34601-0029, o leitor aparenta interpretar um personagem em quase todo o áudio e retomar a voz habitual apenas no final, o que sugere variação na voz do próprio locutor.

Na validação, as 44 falsas não correspondências pertencem ao locutor 118 e comparam o capítulo 47824 com os outros dois, com escore máximo de 0,211. A inspeção auditiva de um arquivo de cada um dos capítulos 47824 e 121721 revelou vozes claramente diferentes, o que indica um provável erro de rótulo na base. Esses pares foram mantidos para não alterar o protocolo após a observação dos resultados. Como 44 de 3.800 pares equivalem a 1,158%, eles explicam por que o EER de validação foi idêntico nas três _seeds_. Além disso, como o EER é o ponto em que FMR e FNMR se igualam, esse piso tende a tornar o limiar mais permissivo do que seria sem esses pares.

#figure(
  image("../results/error_analysis/finetuned_seed42_librispeech_test/overview.png"),
  scope: "parent",
  caption: [Espectrogramas log-Mel das três falsas correspondências e das três falsas não correspondências analisadas, no teste da LibriSpeech (modelo ajustado, _seed_ 42).],
) <fig:erros>

== Avaliação entre bases

Na VCTK, o modelo ajustado manteve um EER inferior ao do pré-treinado, o que indica que o ganho se transfere, ao menos em parte, para uma base não vista. O ponto de operação, por outro lado, transferiu-se mal: com o limiar congelado, a FMR do modelo ajustado foi cerca de duas vezes a do pré-treinado. Uma hipótese é que a mudança de domínio desloque a distribuição de escores e que o limiar mais baixo do modelo ajustado (0,308, contra 0,388) o torne mais sensível a isso.

Nos pares da Dupla 7, as taxas foram maiores para os dois modelos do que no protocolo deste trabalho, o que é compatível com as comparações entre microfones, hipótese que não foi testada isoladamente.

== Limitações

Como as taxas de erro no teste da LibriSpeech são muito baixas, pequenas diferenças de EER correspondem a poucos pares. O teste da VCTK possui apenas 16 locutores, e foram usadas somente três _seeds_, sem testes de significância estatística. O ruído avaliado é apenas branco e gaussiano, e a ablação do VAD foi realizada somente em gravações limpas, não permitindo concluir sobre seu efeito em ambientes ruidosos. Por fim, as hipóteses da análise de erros baseiam-se em inspeção auditiva de poucas amostras. Os pares da Dupla 7 incluem apenas 170 genuínos, de modo que cada um corresponde a 0,59 ponto percentual da FNMR, e as diferenças de EER entre os modelos equivalem a cerca de um par.

= Conclusão

O _fine-tuning_ de uma ResNet-34 pré-treinada reduziu o EER de teste na LibriSpeech de 0,395% para 0,201 $plus.minus$ 0,017%, mas essa melhora na separação não se traduziu em um ponto de operação mais robusto. Com o limiar congelado na validação, o modelo ajustado aceitou mais impostores que o pré-treinado fora da condição de treinamento. Avaliar um sistema biométrico apenas pelo EER pode, portanto, ocultar problemas de uma aplicação real, e o limiar deve ser calibrado em dados representativos da condição de uso.

A análise de erros também mostrou a influência da qualidade dos rótulos e da variação vocal. Como trabalhos futuros, propõe-se investigar a calibração de escores e o aumento de dados com ruído. O código está disponível em #link("https://github.com/SandWoodJones/Biometria") (tag `final-v1.0`).