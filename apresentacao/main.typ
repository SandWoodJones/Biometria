// Apresentação de 10 minutos. Compilar a partir da raiz do repositório:
//   typst watch --root . apresentacao/main.typ

#let azul = rgb("#2a78d6")
#let laranja = rgb("#eb6834")
#let tinta = rgb("#0b0b0b")
#let apagado = rgb("#52514e")

#set document(title: [Verificação de Locutor com ResNet-34 e _Fine-Tuning_], author: ("Raul Rodrigues", "Paulo Henrique"))
#set page(paper: "presentation-16-9", margin: (x: 1.6cm, top: 1.3cm, bottom: 1.1cm))
#set text(lang: "pt", region: "br", font: "TeX Gyre Heros", size: 20pt, fill: tinta, hyphenate: false)
#set par(leading: 0.6em)
#set list(marker: text(fill: azul)[•], spacing: 0.9em)
// Termos compostos nunca quebram no hífen nem se separam da unidade.
#show regex("pré-treinad[ao]s?|log-Mel|\d+ dB"): it => box(it)

// Texto ainda não aprovado, visível no PDF.
#let escrever(body) = block(
  width: 100%,
  inset: 10pt,
  fill: rgb("#fff4e5"),
  stroke: (left: 3pt + laranja),
  text(size: 16pt, fill: rgb("#7a3410"))[*A escrever:* #body],
)

#let numero = counter("slide")

// Um slide com título e conteúdo; o texto do slide é lido em voz alta.
#let slide(titulo, body) = {
  numero.step()
  page(footer: context {
    set text(size: 12pt, fill: apagado)
    [Verificação de Locutor com ResNet-34 e _Fine-Tuning_ #h(1fr) #numero.display() / #numero.final().first()]
  })[
    #if titulo != none {
      text(size: 30pt, weight: "bold", titulo)
      v(-0.5em)
      line(length: 100%, stroke: 2pt + azul)
      v(0.3em)
    }
    #body
  ]
}

// 1 — Raul
#slide(none)[
  #text(size: 32pt, weight: "bold")[Verificação de Locutor com ResNet-34 e #box[_Fine-Tuning_]]
  #v(-0.4em)
  #line(length: 100%, stroke: 2pt + azul)
  #v(-0.3em)
  #text(size: 20pt)[Raul Rodrigues e Paulo Henrique]
  #h(1fr)
  #text(size: 16pt, fill: apagado)[Biometria — UTFPR, Câmpus Ponta Grossa]
  #v(1fr)
  #block(inset: (left: 14pt), stroke: (left: 3pt + azul))[
    Na verificação de locutor, o sistema recebe duas gravações de voz e decide se elas são da mesma pessoa.

    Avaliamos uma ResNet-34 pré-treinada, antes e depois de ajustá-la na LibriSpeech.
  ]
  #v(0.6em)
  #text(fill: azul, weight: "bold")[Pergunta:] um EER menor garante um sistema melhor na prática?
  #v(1fr)
]

// 2 — Raul
#let cartao(titulo, body) = block(
  width: 100%,
  height: 100%,
  inset: 14pt,
  radius: 6pt,
  fill: azul.lighten(90%),
  stroke: (left: 4pt + azul),
)[#text(weight: "bold", fill: azul, titulo) \ #body]

#slide([Dados e protocolo])[
  #grid(
    columns: (1fr, 1fr),
    rows: 4cm,
    gutter: 0.8cm,
    cartao([LibriSpeech])[251 locutores lendo audiolivros. Usada no ajuste e no teste.],
    cartao([VCTK])[110 locutores gravados em estúdio. Usada só no teste, como base que o modelo nunca viu.],
  )
  #v(0.6em)
  - Os locutores foram divididos em treino, validação e teste, sem nenhum locutor em comum.
  - Cada teste compara milhares de pares: 3.800 genuínos e 38.000 impostores na LibriSpeech.
  - O limiar de decisão foi escolhido na validação e congelado antes de qualquer teste.
]

// 3 — Raul
#slide([_Pipeline_ e modelo])[
  #grid(
    columns: (1.15fr, 1fr),
    gutter: 0.8cm,
    align: horizon,
    text(size: 18pt)[
      Trechos com energia muito abaixo da parte mais alta do áudio são tratados como silêncio e removidos. O resto vira um espectrograma #box[log-Mel].

      A ResNet-34, pré-treinada na VoxCeleb2, transforma o espectrograma em um vetor de 256 números: o _embedding_.

      Ajustamos a rede na LibriSpeech, repetindo o treino com três _seeds_.

      Duas gravações são comparadas pela similaridade de cosseno entre seus _embeddings_.
    ],
    image("../results/pipeline.png", width: 100%),
  )
]

// 4 — Raul
#slide([Resultado na LibriSpeech])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    align: horizon,
    [
      #align(center, text(size: 30pt, weight: "bold")[EER #text(fill: apagado)[0,395%] → #text(fill: azul)[0,201%]])
      #v(0.3em)
      No teste da LibriSpeech, o ajuste reduziu o EER de 0,395% para 0,201%, na média das três _seeds_.

      Com o limiar congelado, o modelo ajustado errou menos dos dois lados: aceitou menos impostores e rejeitou menos genuínos.

      No gráfico, os escores de genuínos e de impostores quase não se sobrepõem.
    ],
    image("../results/finetuned_seed42/librispeech_test/distributions.png", width: 100%),
  )
]

// 5 — Paulo
#slide([Avaliação entre bases])[
  #align(center, text(size: 18pt, table(
    columns: 5,
    align: (left, right, right, right, right),
    inset: (x: 12pt, y: 8pt),
    stroke: (x, y) => if y == 1 or y == 2 { (top: 1pt + apagado) },
    table.header(
      [], table.cell(colspan: 2, align: center)[*EER*], table.cell(colspan: 2, align: center)[*FMR*],
      [], [Pré-treinado], [Ajustado], [Pré-treinado], [Ajustado],
    ),
    [VCTK, nossos pares], [0,687%], [0,581%], [1,825%], text(fill: laranja, weight: "bold")[3,938%],
    [VCTK, pares da Dupla 7], [0,724%], [0,611%], [3,020%], text(fill: laranja, weight: "bold")[4,393%],
  )))
  #v(0.6em)
  Na VCTK, que o modelo nunca viu, o ajuste também reduziu o EER.

  Mas, com o limiar congelado, o modelo ajustado aceitou cerca de duas vezes mais impostores. A FMR é a fração de impostores aceitos.

  Nos pares da Dupla 7, que misturam dois microfones, o padrão se repetiu.
]

// 6 — Paulo
#slide([Robustez a ruído])[
  #grid(
    columns: (1fr, auto),
    gutter: 0.8cm,
    align: horizon,
    text(size: 18pt)[
      Adicionamos ruído branco às gravações em três níveis de SNR, que mede quanto a voz é mais forte que o ruído: 20 dB (ruído fraco), 10 dB e 0 dB (ruído tão forte quanto a voz).

      O EER piorou nos dois modelos, mas o ajustado continuou melhor em todos os níveis.

      Com o limiar congelado, porém, o modelo ajustado aceitou #text(fill: laranja, weight: "bold")[52,8%] dos impostores no ruído mais forte, contra 10,8% do pré-treinado. Esse valor variou muito entre as _seeds_.
    ],
    align(center, image("../results/summary/robustness_vertical.png", height: 11cm)),
  )
]

// 7 — Paulo
#slide([Análise de erros])[
  #grid(
    columns: (1.3fr, 1fr),
    gutter: 0.8cm,
    align: horizon,
    image("../results/error_analysis/finetuned_seed42_librispeech_test/overview.png", width: 100%),
    text(size: 18pt)[
      Nas falsas correspondências, a maioria dos pares é de locutores do mesmo gênero, com vozes parecidas: 216 de 237.

      As falsas não correspondências do teste são todas do locutor 6209. Em um dos capítulos, o leitor imita a voz de um personagem.

      Na validação, os erros do locutor 118 envolvem vozes claramente diferentes: provavelmente um erro de rótulo na própria base.
    ],
  )
]

// 8 — Paulo
#slide([Conclusão])[
  O ajuste reduziu o EER em todas as condições testadas.

  Mas, com o limiar congelado, ele aceitou mais impostores fora da condição de treino: em outra base e com ruído.

  #v(0.4em)
  #block(width: 100%, inset: 14pt, radius: 6pt, fill: azul.lighten(90%), stroke: (left: 4pt + azul))[
    Um EER menor não garante um sistema melhor na prática. O limiar precisa ser calibrado nas condições reais de uso.
  ]
  #v(1fr)
  #text(size: 16pt, fill: apagado)[*Código:* github.com/SandWoodJones/Biometria (tag `final-v1.0`)]
]
