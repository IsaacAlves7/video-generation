# 🎥🎬 Video Generation Landscape
Uma coisa importante para começar: modelos como o **Seedance 2.0**, o **Veo**, o **Sora**, o **Kling** ou o **Wan** normalmente não são apenas uma "LLM que gera vídeo". Na verdade, eles são sistemas multimodais compostos por vários modelos especializados trabalhando juntos. A LLM participa principalmente da compreensão do prompt, mas a geração visual em si costuma ser realizada por arquiteturas de difusão, transformers multimodais e módulos temporais específicos para vídeo.

O que o Seedance 2.0 é por baixo, é um **Video Diffusion Transformer** (similar ao DiT) com geração conjunta de áudio+vídeo. Para replicar algo nesse nível você precisaria de:

- ~50.000–500.000 horas de vídeo rotulado de alta qualidade
- Clusters de A100/H100 por **semanas ou meses**
- Time de dezenas de pesquisadores
- Orçamento estimado: **$5M–$50M** só em compute

Isso está fora do alcance de uma pessoa ou equipe pequena, antes de qualquer código: Você não pode criar um modelo "baseado no Seedance 2.0" — os pesos, arquitetura interna e dados de treinamento são propriedade fechada do ByteDance. Não há como clonar, fazer fine-tune ou destilação do Seedance 2.0 legalmente, porque os pesos nunca foram publicados.

O que você realmente quer é provavelmente isso: rodar um modelo de geração de vídeo open-source de graça, no seu próprio servidor, sem pagar API nenhuma. Isso é totalmente viável. Os melhores candidatos hoje:

| Modelo | Qualidade | VRAM mínima | Licença |
|---|---|---|---|
| **Wan 2.1** (Alibaba) | ⭐⭐⭐⭐⭐ | 8 GB | Apache 2.0 ✅ |
| **HunyuanVideo** (Tencent) | ⭐⭐⭐⭐⭐ | 24 GB | Open ✅ |
| **CogVideoX-5B** (Zhipu) | ⭐⭐⭐⭐ | 16 GB | Apache 2.0 ✅ |
| **Open-Sora** | ⭐⭐⭐ | 12 GB | Apache 2.0 ✅ |

O **Wan 2.1** é o mais próximo em qualidade do Seedance 2.0 e roda em GPUs acessíveis (RTX 3080/4070). É viável construir a mesma aplicação FastAPI + React, mas apontando para o **Wan 2.1 rodando 100% local**, sem nenhuma API externa.

## DiT - Diffusion Transformer
O **DiT (Diffusion Transformer)** é uma das evoluções mais importantes dos modelos de difusão modernos e está diretamente relacionado ao surgimento de geradores de vídeo extremamente avançados como Sora, Veo, Wan, Kling e outros.

Para entender o DiT, primeiro é preciso lembrar como os modelos de difusão nasceram.

Os primeiros modelos de difusão usavam predominantemente CNNs (Convolutional Neural Networks), especialmente arquiteturas chamadas **U-Net**. Durante anos, o pipeline era algo parecido com:

```text
Prompt
   ↓
Encoder de Texto
   ↓
U-Net
   ↓
Difusão
   ↓
Imagem
```

O U-Net era responsável por olhar para o ruído e decidir como removê-lo passo a passo.

Funcionava muito bem para imagens.

Mas quando começaram a surgir modelos gigantescos, vídeos longos e cenas complexas, apareceram limitações.

As CNNs são excelentes para detectar padrões locais.

Por exemplo:

* bordas;
* texturas;
* pequenas regiões da imagem.

Mas elas têm mais dificuldade em entender relações globais.

Imagine uma cena:

```text
Castelo
           ↓
Montanha
           ↓
Dragão voando
```

O modelo precisa compreender a relação entre todos esses elementos simultaneamente.

É aí que entram os Transformers.

Os Transformers revolucionaram primeiro o NLP.

Depois foram adaptados para visão computacional.

Primeiro vieram os:

* Vision Transformers (ViT)
* Swin Transformers
* Video Transformers

Depois alguém teve a ideia:

> "E se substituirmos o U-Net inteiro por um Transformer?"

Nasceu o DiT.

Em vez de processar a imagem usando convoluções, o modelo divide a imagem em pequenos blocos chamados patches.

Imagine uma imagem:

```text
████████
████████
████████
████████
```

Ela é quebrada em pedaços:

```text
[Patch 1]
[Patch 2]
[Patch 3]
...
```

Cada patch vira um token.

Exatamente como uma palavra vira um token em uma LLM.

É por isso que existe uma forte conexão entre:

```text
LLMs
Vision Transformers
DiTs
```

Todos trabalham sobre tokens.

Um GPT pode receber:

```text
Batman
walks
in
Gotham
```

Um DiT recebe:

```text
Patch 1
Patch 2
Patch 3
Patch 4
...
```

A magia acontece na Self-Attention.

O Transformer consegue perguntar:

> "Quais partes da imagem são importantes para entender esta outra parte?"

Isso permite enxergar dependências globais.

Por exemplo:

```text
Olho esquerdo
        ↔
Olho direito

Rosto
        ↔
Corpo

Pessoa
        ↔
Sombra
```

Tudo ao mesmo tempo.

Não existe mais aquela limitação local típica das convoluções.

Nos vídeos a coisa fica ainda mais interessante.

Imagine um tensor:

```text
Largura
Altura
Tempo
```

Ou:

```text
X
Y
T
```

Agora os patches também possuem dimensão temporal.

O DiT pode observar:

```text
Frame 1
Frame 2
Frame 3
Frame 4
```

simultaneamente.

Ele consegue entender:

* movimento;
* velocidade;
* direção;
* consistência temporal.

É uma das razões pelas quais os vídeos modernos parecem muito mais estáveis.

Antes:

```text
Frame 1 → Pessoa A

Frame 2 → Pessoa B

Frame 3 → Pessoa C
```

O personagem mudava constantemente.

Com Transformers temporais e DiTs:

```text
Frame 1 → Pessoa A

Frame 2 → Pessoa A

Frame 3 → Pessoa A
```

A identidade é preservada com muito mais eficiência.

Outra vantagem é a escalabilidade.

Existe uma observação interessante na pesquisa atual:

> Quanto maior o Transformer, melhor o DiT costuma escalar.

Isso lembra muito o comportamento das LLMs.

Você aumenta:

* parâmetros;
* dados;
* GPUs;
* tempo de treinamento.

E o modelo continua melhorando.

Por isso muitos laboratórios abandonaram progressivamente as U-Nets para modelos baseados em Transformer.

O pipeline moderno costuma ser algo parecido com:

```text
Prompt
   ↓
Text Encoder
   ↓
Embeddings
   ↓
DiT
   ↓
Difusão Latente
   ↓
Decoder
   ↓
Imagem ou Vídeo
```

No caso de um gerador como o Seedance 2.0, você pode imaginar algo ainda mais sofisticado:

```text
Prompt
   ↓
LLM
   ↓
Entendimento semântico
   ↓
Embeddings multimodais
   ↓
DiT Espaço-Temporal
   ↓
Difusão de vídeo
   ↓
Refinamento
   ↓
Upscaling
   ↓
Vídeo Final
```

O motivo de tanta gente no mercado de IA estar falando de DiTs atualmente é que eles representam para a geração visual algo semelhante ao que os Transformers representaram para o NLP.

Da mesma forma que o GPT substituiu arquiteturas antigas de linguagem, os DiTs estão gradualmente substituindo arquiteturas tradicionais de difusão baseadas em convolução.

Por isso, quando você vê vídeos extremamente consistentes gerados por modelos de última geração, existe uma grande chance de haver um Diffusion Transformer fazendo boa parte do trabalho pesado por trás dos panos.

Deixa eu clarificar como o DiT realmente funciona: A divisão real no DiT para vídeo: O **conditioning** (o que você passa como entrada para guiar a geração) é que varia, não o tipo de modelo. Um mesmo modelo DiT pode aceitar múltiplos tipos de condicionamento:

Tipos de condicionamento:

**Text conditioning (T2V)**: O prompt de texto é codificado por um encoder de linguagem (T5, CLIP, etc.) e injetado no DiT via cross-attention. O modelo aprende a associar descrições textuais com padrões visuais.

**Image conditioning (I2V)**: Uma imagem de referência é codificada por um VAE e injetada como contexto visual. O modelo aprende a "animar" ou "continuar" a partir daquela imagem.

**Ambos juntos (multimodal)**: É o que o Seedance 2.0, Wan 2.1 e HunyuanVideo fazem — o mesmo backbone DiT recebe texto + imagem + áudio simultâneamente via diferentes mecanismos de atenção.

Como o DiT processa tudo isso:

```
Entrada bruta
     │
     ▼
  VAE Encoder  ───────────────────────────────────┐
(comprime vídeo/imagem em espaço latente)         │
     │                                            │
     ▼                                            ▼
Latent Tokens                           Conditioning Tokens
(o vídeo virou patches)           (texto via T5, imagem via CLIP)
     │                                            │
     └──────────────┬─────────────────────────────┘
                    ▼
             DiT Blocks
        (self-attention + cross-attention)
                    │
                    ▼
             VAE Decoder
        (latente → vídeo final)
```

Os dois componentes que você treina separadamente

| Componente | O que faz | Treina separado? |
|---|---|---|
| **VAE** | Comprime e reconstrói frames | Sim, geralmente pré-treinado |
| **DiT backbone** | Aprende a fazer diffusion guiado | Sim, é o núcleo |
| **Text encoder** | Entende prompts | Geralmente congelado (T5/CLIP pré-treinados) |

Então quando você vê "modelo T2V" e "modelo I2V" no Wan ou CogVideoX, na prática são **pesos diferentes do mesmo backbone DiT**, treinados com estratégias de conditioning diferentes — não arquiteturas separadas.

Qual parte você quer aprofundar — a arquitetura do DiT em si, o VAE, ou como estruturar o pipeline de treinamento?

## O que é realisticamente possível fazer
Existem 3 caminhos viáveis dependendo do seu objetivo real:

**Caminho 1 — Fine-tune de um modelo open-source** *(semanas, ~$500–2000 em GPU)*: Pegar o Wan 2.1 ou CogVideoX e especializar em um estilo/domínio específico com seus próprios dados. Resultado: modelo seu, customizado, rodando local.

**Caminho 2 — Construir um modelo pequeno do zero** *(meses, viável com 1 GPU)*: Implementar um Video Diffusion Transformer simples com PyTorch, treinar em um dataset público como WebVid ou Panda-70M. Qualidade bem abaixo do Seedance, mas é **seu** modelo, você entende cada camada.

**Caminho 3 — Distilação de conhecimento de open-sources** *(meses)*: Usar modelos open como "professores" para treinar um modelo menor e mais eficiente. Legalmente viável se os professores forem modelos com licença permissiva.

Qual desses caminhos faz mais sentido para você? E qual é o objetivo final — aprender a fundo, ter um modelo para produção, ou especializar em um domínio específico? Isso muda completamente o que vale a pena construir.

Quando você instala um sistema open source de geração de vídeo localmente, como um pipeline baseado em Diffusers, Wan ou CogVideoX, o processo começa muito antes da geração. Primeiro você precisa do ambiente: Python, CUDA, drivers NVIDIA, PyTorch, bibliotecas de inferência e os pesos do modelo. Os pesos são os arquivos gigantescos que armazenam o conhecimento aprendido durante o treinamento. Dependendo do modelo, eles podem ocupar dezenas ou centenas de gigabytes.

Após a instalação, o modelo não "sabe gerar vídeos" da forma como um programador escreveria um algoritmo tradicional. O que ele possui é uma enorme quantidade de parâmetros numéricos aprendidos durante o treinamento. Durante esse treinamento, bilhões de imagens, vídeos e descrições textuais foram apresentados ao sistema. O modelo aprendeu relações estatísticas extremamente complexas entre palavras, conceitos, objetos, movimentos, iluminação, câmeras, estilos artísticos, física aparente e comportamento visual.

Quando você escreve um prompt como:

> "Um cavaleiro medieval caminhando em uma floresta ao amanhecer, câmera cinematográfica, neblina suave."

A primeira etapa geralmente passa por um encoder de texto. Muitas arquiteturas modernas usam algo semelhante a uma LLM compacta ou um encoder derivado de transformers. O texto é transformado em embeddings.

Um embedding é uma representação vetorial.

Em vez de armazenar:

```text
cavaleiro
floresta
amanhecer
```

o sistema converte tudo em milhares de números.

Algo conceitualmente parecido com:

```text
[0.1532, -0.7821, 0.9911, ...]
```

Esses vetores carregam significado semântico.

O modelo aprende que:

```text
rei ≈ rainha
homem ≈ mulher
cachorro ≈ lobo
```

ocupam regiões próximas do espaço vetorial.

O prompt inteiro vira uma estrutura matemática gigantesca.

A partir daí começa a parte mais interessante.

Muita gente imagina que o modelo desenha quadro por quadro.

Na realidade, os modelos modernos de difusão fazem quase o oposto.

Eles começam com ruído puro.

Literalmente algo parecido com televisão sem sinal.

Imagine um tensor multidimensional preenchido por valores aleatórios.

O estado inicial se parece com:

```text
█████████████
█████████████
█████████████
█████████████
```

Sem significado algum.

O trabalho do modelo é remover o ruído gradualmente.

Por isso o nome "difusão".

Durante o treinamento, o modelo aprendeu duas tarefas:

Adicionar ruído.

Remover ruído.

Na inferência ele executa apenas a segunda.

O processo ocorre em dezenas ou centenas de passos.

Em cada passo ele pergunta:

> "Com base no prompt e no estado atual, qual parte deste ruído parece errada?"

Então corrige.

Depois corrige novamente.

Depois corrige novamente.

Até que o ruído começa a adquirir forma.

Primeiro surgem manchas.

Depois silhuetas.

Depois objetos.

Depois detalhes.

Em um gerador de imagens, o resultado final seria uma única imagem.

Em vídeo isso não basta.

O sistema precisa resolver um problema muito mais difícil:

Consistência temporal.

Imagine gerar:

```text
Frame 1
Frame 2
Frame 3
Frame 4
```

independentemente.

O cavaleiro poderia ter:

* olhos diferentes;
* armadura diferente;
* cabelo diferente;
* posição impossível.

O vídeo pareceria um pesadelo psicodélico.

Então os modelos modernos trabalham em um espaço espaço-temporal.

Em vez de pensar:

```text
altura
largura
```

eles pensam:

```text
altura
largura
tempo
```

O vídeo é tratado como um bloco tridimensional.

Conceitualmente:

```text
[x, y, t]
```

onde t representa o tempo.

A geração ocorre simultaneamente sobre múltiplos quadros.

O modelo aprende que:

* objetos devem permanecer os mesmos;
* iluminação deve ser coerente;
* movimentos devem ser suaves;
* personagens devem manter identidade.

É aqui que entram transformers temporais e attention temporal.

A atenção temporal funciona como uma memória.

Quando o modelo está gerando um quadro futuro, ele consegue olhar para quadros anteriores.

Algo como:

```text
Frame 20
↓
consulta
↓
Frame 19
Frame 18
Frame 17
```

Isso evita mudanças bruscas.

Nos modelos mais modernos, como Seedance 2.0 e Sora, existe ainda uma compreensão implícita de física.

Não é física real.

É física estatística aprendida.

O modelo observou tantos vídeos que aprendeu padrões como:

* gravidade;
* aceleração;
* colisões;
* reflexos;
* comportamento da água;
* fumaça;
* tecidos.

Ele não resolve as equações de Newton explicitamente.

Ele aprendeu visualmente.

Da mesma forma que um ser humano consegue prever aproximadamente como uma bola cairá sem calcular integrais.

Outro componente importante é o VAE (Variational Autoencoder).

Vídeos possuem uma quantidade absurda de pixels.

Gerar tudo diretamente seria inviável.

Então o vídeo é comprimido para um espaço latente.

Pense nisso como:

```text
Vídeo original
↓
Compressão inteligente
↓
Espaço latente
↓
Difusão
↓
Reconstrução
↓
Vídeo final
```

Em vez de trabalhar com milhões de pixels diretamente, o modelo trabalha em uma representação comprimida muito menor.

Isso reduz drasticamente o custo computacional.

Nos sistemas mais avançados, a geração também envolve múltiplos estágios.

Primeiro surge um vídeo de baixa resolução.

Depois ocorre upscaling.

Depois refinamento.

Depois estabilização temporal.

Depois correções de detalhes.

O vídeo que você recebe normalmente passou por várias redes neurais diferentes.

Quando o vídeo termina de ser gerado, ele ainda não está necessariamente em formato MP4.

Internamente existe apenas uma sequência de tensores.

O sistema converte esses tensores para imagens.

Depois as imagens são codificadas em vídeo usando codecs como H.264 ou H.265.

Somente então surge o arquivo final.

Por baixo dos panos, um prompt aparentemente simples como:

> "Batman caminhando em Gotham sob chuva intensa à noite"

pode disparar bilhões de operações matemáticas.

Milhares de multiplicações matriciais.

Centenas de camadas transformer.

Dezenas de passos de difusão.

Gigabytes de dados transitando pela memória da GPU.

E tudo isso acontece porque o modelo aprendeu uma representação matemática extremamente compacta do mundo visual a partir de enormes volumes de vídeos durante o treinamento.

A parte mais impressionante é que o modelo não possui uma biblioteca de vídeos escondida que ele mistura. Ele não procura um vídeo de Batman e outro de chuva para colar um no outro. O que ele faz é gerar uma nova sequência visual a partir das distribuições estatísticas aprendidas. Cada quadro é sintetizado matematicamente naquele momento, condicionado pelo prompt e pelos estados temporais anteriores. É por isso que duas execuções do mesmo prompt podem produzir vídeos diferentes, mas ainda coerentes com a descrição fornecida.

## News:
- https://www.fanaticalfuturist.com/2018/04/ai-can-create-videos-out-of-thin-air-using-just-text-as-an-input/
- https://www.techinasia.com/gliacloud-uses-artificial-intelligence-to-automatically-turn-text-into-video
- https://github.com/vita-epfl/Stable-Video-Infinity

## Reddit Posts:

https://www.reddit.com/r/MachineLearning/comments/9wk188/p_biggan_generators_on_tf_hub_with_colab_demo/

## Samples from github:

| Samples       | Pretrained Models       | Code  | Paper  | Output Quality | License |
| ------------- |:-----------------------:| -----:| ------:|---------------:|-------: |
|[Memoji](https://www.youtube.com/watch?v=CjqERCCD4iM)|[Model]()|[Code]()|[](https://arxiv.org/abs/)| A | Non Commercial CC |
|[Deep Fakes demo](https://www.youtube.com/watch?v=VXZlq70jHvw)|[(Download Pretrained model)](https://anonfile.com/p7w3m0d5be/face-swap.zip)|[Code]()|[](https://arxiv.org/abs/)| A |--|
|[VideoGAN](https://www.youtube.com/watch?v=Pt1W_v-yQhw)|[Download Model](https://drive.google.com/file/d/0B-xMJ5CYz_F9QS1BTE5yWl9aUWs/view?usp=sharing)|[Code](https://github.com/cvondrick/videogan)|[Tinyvideo](http://www.cs.columbia.edu/~vondrick/tinyvideo/)|--|--|
|[Adversarial Video Generation](https://github.com/dyelax/Adversarial_Video_Generation#results-and-comparison)|[Download Model](https://drive.google.com/open?id=0Byf787GZQ7KvR2JvMUNIZnFlbm8)|[Code](https://github.com/dyelax/Adversarial_Video_Generation)|[1511.05440](https://arxiv.org/abs/1511.05440)|--|--|
|[Improved VideoGAN](https://bernhard2202.github.io/ivgan/index.html)|--|[Code](https://github.com/bernhard2202/improved-video-gan)|[1711.11453](https://arxiv.org/abs/1711.11453)|--|--|

## Work in progress:
- https://github.com/alexlee-gk/video_prediction
- https://github.com/liuziwei7/voxel-flow
- https://github.com/Yunbo426/predrnn-pp
- https://github.com/mabirck/Video_GAN_Sonic
- https://github.com/xzr12/PredCNN

If I missed your output sample/demo in this consolidation, just add and send a pull request. I will be more than happy to add it. Thanks!

## Product Demos:
- [Nvidea](https://www.youtube.com/watch?v=kSLJriaOumA)(official)
- [Deep Fakes demo](https://www.youtube.com/watch?v=VXZlq70jHvw)

## Arxiv-sanity
