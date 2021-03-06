This repository contains [MXNet](https://mxnet.incubator.apache.org/) implementations of models from various papers. My plan is to first focus on generative image models.

Currently, I have
- [Variational autoencoder](vae/) from [1] (Completed)
- [DRAW](draw/) from [2] (Completed)
- [ConvDRAW](convdraw/) from [3] (Completed)
- [Generative Query Network](gqn/) from [4] (Completed)

See the README file under its folder for instructions on running a model.

The code in this repository was tested under Python 3.5.2. You can find all the required python packages in [`requirements.txt`](requirements.txt).


**References**

[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114

[2] K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra, “DRAW: A Recurrent Neural Network For Image Generation,” arXiv:1502.04623 [cs], Feb. 2015.

[3] K. Gregor, F. Besse, D. J. Rezende, I. Danihelka, and D. Wierstra, “Towards Conceptual Compression” arXiv:1604.08772 [stat.ML], Apr. 2015.

[4] S. M. A. Eslami, D. J. Rezende et al. "Neural scene representation and rendering" Science. 2018

Please open up an issue if there is a particular model you'd like me to implement.

