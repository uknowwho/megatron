An implementation of the transformer architecture, loosely based on [[1](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)], aptly called Megatron. Megatron uses Self-Attention, Layer Normalization, Residual Connections, and Multi-Layer Perceptrons. Built using [this spectacular video](https://www.youtube.com/watch?v=kCc8FmEb1nY). Trained on [[2](https://arxiv.org/abs/1908.02322)]. 

A demo is available [here](https://colab.research.google.com/drive/1TvI3j0vG8jHtNoXeAA6qAeJshZlanbhC). 

The best performing model is included under Megatron-9M.pt

![https://raw.githubusercontent.com/uknowwho/megatron/main/Megatron-9M-lossprogression.png](https://github.com/uknowwho/megatron/blob/main/Megatron-9M-lossprogression.png)

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30, pp 6000-6010. (2017)

[2] Chia-Lun Yeh, Babak Loni, Mariëlle Hendriks, Henrike Reinhardt and Anne Schuth: DpgMedia2019: A Dutch News Dataset for Partisanship Detection. arXiv: arXiv:1908.02322 [cs]
