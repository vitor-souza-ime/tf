# Transfer Learning com ResNet18 e CIFAR-10

Este repositório contém uma implementação de **Transfer Learning** utilizando a arquitetura **ResNet18** pré-treinada no **ImageNet**, aplicada a um problema de classificação binária com as classes "gato" e "cachorro" do conjunto **CIFAR-10** (Krizhevsky, 2009). O objetivo é demonstrar empiricamente o impacto do transfer learning em cenários de dados escassos.

## Conteúdo

- `main.py` — Script principal que realiza:
  - Pré-processamento das imagens (redimensionamento e normalização);
  - Configuração de dois modelos ResNet18: do zero e com transfer learning;
  - Treinamento de ambos os modelos por 5 épocas usando Adam;
  - Avaliação de acurácia e perda;
  - Geração de gráficos comparativos com Matplotlib.

## Requisitos

- Python 3.8 ou superior
- PyTorch e torchvision
- Matplotlib
- Numpy
- GPU com CUDA (opcional, mas recomendado)

Instalação rápida com pip:

```bash
pip install torch torchvision matplotlib numpy
````

## Estrutura do Projeto

```
tf/
├── main.py          # Código principal
├── README.md        # Este arquivo
```

## Execução

Para treinar e avaliar os modelos:

```bash
python main.py
```

O script exibirá no terminal o progresso do treinamento, incluindo perda e acurácia por época, e gerará gráficos comparando os resultados dos dois modelos.

## Metodologia

* O **modelo do zero** é inicializado sem pesos pré-treinados, com todos os parâmetros treináveis.
* O **modelo de transfer learning** utiliza pesos pré-treinados no ImageNet (He et al., 2016; Russakovsky et al., 2015), com todas as camadas convolucionais congeladas e apenas a camada final treinável.
* Função de perda: Cross-Entropy Loss
* Otimizador: Adam (Kingma e Ba, 2015)
* Dataset: CIFAR-10, apenas classes "gato" (0) e "cachorro" (1), treinamento limitado a 500 imagens, teste com 2.000 imagens.
* Pré-processamento: redimensionamento para 224×224 e normalização com média `[0.485, 0.456, 0.406]` e desvio padrão `[0.229, 0.224, 0.225]`.

## Referências

* PAN, S. J.; YANG, Q. *A Survey on Transfer Learning*. IEEE Transactions on Knowledge and Data Engineering, 2010.
* YOSINSKI, J. et al. *How transferable are features in deep neural networks?* NIPS, 2014.
* HE, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
* RUSSAKOVSKY, O. et al. *ImageNet Large Scale Visual Recognition Challenge*. IJCV, 2015.
* KRIZHEVSKY, A. *Learning Multiple Layers of Features from Tiny Images*. 2009.
* KINGMA, D. P.; BA, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015.

## Licença

Este projeto está licenciado sob a MIT License.
