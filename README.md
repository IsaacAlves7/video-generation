# 🎥🎬 Video Generation Landscape
https://piapi.ai/docs/seedance-api/seedance-2

Uma coisa importante para começar: modelos como o **Seedance 2.0**, o **Veo**, o **Sora**, o **Kling** ou o **Wan** normalmente não são apenas uma "LLM que gera vídeo". Na verdade, eles são sistemas multimodais compostos por vários modelos especializados trabalhando juntos. A LLM participa principalmente da compreensão do prompt, mas a geração visual em si costuma ser realizada por arquiteturas de difusão, transformers multimodais e módulos temporais específicos para vídeo.

O que o Seedance 2.0 é por baixo, é um **Video Diffusion Transformer** (similar ao DiT) com geração conjunta de áudio+vídeo. Para replicar algo nesse nível você precisaria de:

- ~50.000–500.000 horas de vídeo rotulado de alta qualidade
- Clusters de A100/H100 por **semanas ou meses**
- Time de dezenas de pesquisadores
- Orçamento estimado: **$5M–$50M** só em compute

Isso está fora do alcance de uma pessoa ou equipe pequena.

## O que é realisticamente possível fazer

Existem 3 caminhos viáveis dependendo do seu objetivo real:

**Caminho 1 — Fine-tune de um modelo open-source** *(semanas, ~$500–2000 em GPU)*
Pegar o Wan 2.1 ou CogVideoX e especializar em um estilo/domínio específico com seus próprios dados. Resultado: modelo seu, customizado, rodando local.

**Caminho 2 — Construir um modelo pequeno do zero** *(meses, viável com 1 GPU)*
Implementar um Video Diffusion Transformer simples com PyTorch, treinar em um dataset público como WebVid ou Panda-70M. Qualidade bem abaixo do Seedance, mas é **seu** modelo, você entende cada camada.

**Caminho 3 — Distilação de conhecimento de open-sources** *(meses)*
Usar modelos open como "professores" para treinar um modelo menor e mais eficiente. Legalmente viável se os professores forem modelos com licença permissiva.

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


### Support:

If you want the good work to continue please support us on

* [PAYPAL](https://www.paypal.me/ishandutta2007)
* [BITCOIN ADDRESS: 3LZazKXG18Hxa3LLNAeKYZNtLzCxpv1LyD](https://www.coinbase.com/join/5a8e4a045b02c403bc3a9c0c)
