# Unbiased minibatch Sinkhorn GAN
Official Python3 implementation of the paper of [Minibatch optimal transport distances; analysis and applications](https://arxiv.org/pdf/2101.01792.pdf)

### Abstract
Optimal transport distances have become a classic tool to compare probability distributions and have found many applications in machine learning.
Yet, despite recent algorithmic developments, their complexity prevents their direct use on large scale datasets. To overcome this challenge, a common workaround is to compute these distances on minibatches {\em i.e.} to average the outcome of several smaller optimal transport problems. We propose in this paper an extended analysis of this practice, which effects were previously studied in restricted cases. We first consider a large variety of Optimal Transport kernels. We notably argue that the minibatch strategy comes with appealing properties such as unbiased estimators, gradients and a concentration bound around the expectation, but also with limits: the minibatch OT is not a distance. To recover some of the lost distance axioms, we introduce a debiased minibatch OT function and study its statistical and optimisation properties. Along with this theoretical analysis, we also conduct empirical experiments on gradient flows, generative adversarial networks (GANs) or color transfer that highlight the practical interest of this strategy.

### How to cite
This paper is an extension of an AISTATS 2020 conference paper [Learning with minibatch Wasserstein: asymptotic and gradient properties](https://arxiv.org/abs/1910.04091).

If you use this toolbox in your research or minibatch Wasserstein and find them useful, please cite "Minibatch optimal transport distances; analysis and applications" and "Learning with minibatch Wasserstein: asymptotic and gradient properties" using the following bibtex reference:

```
@misc{fatras2021minibatch,
      title={Minibatch optimal transport distances; analysis and applications}, 
      author={Kilian Fatras and Younes Zine and Szymon Majewski and Rémi Flamary and Rémi Gribonval and Nicolas Courty},
      year={2021},
      eprint={2101.01792},
      archivePrefix={arXiv},
}
```

```
@InProceedings{fatras2019learnwass,
author    = {Fatras, Kilian and Zine, Younes and Flamary, Rémi and Gribonval, Rémi and Courty, Nicolas},
title     = {Learning with minibatch Wasserstein: asymptotic and gradient properties},
booktitle = {AISTATS},
year      = {2020}
}
```

### Blog post

We also wrote a [medium blog post](https://medium.com/p/learning-with-minibatch-wasserstein-d87dcf52efb5?source=email-d0d7857135bb--writer.postDistributed&sk=4c30efd3442780edf7ca140080557476), feel free to ask if any question.

### Prerequisites

* Python3 (3.7.3)
* PyTorch (1.6.0)
* POT (0.6.0)
* Numpy (1.16.4)
* Scipy (1.2.0)
* argparse (1.1)
* os
* CUDA


### What is included ?

* Unbiased Sinkhorn GAN experiment


### Authors

* [Kilian Fatras](https://kilianfatras.github.io/)
* [Younès Zine](https://www.linkedin.com/in/youn%C3%A8s-zine-7abb68149/?originalSubdomain=fr)
* [Szymon Majewski](https://scholar.google.com/citations?user=xxTq-sYAAAAJ&hl=pl)
* [Rémi Flamary](http://remi.flamary.com/)
* [Rémi Gribonval](http://people.irisa.fr/Remi.Gribonval/)
* [Nicolas Courty](https://github.com/ncourty)


## References

[1] Flamary Rémi and Courty Nicolas [POT Python Optimal Transport library](https://github.com/rflamary/POT)
