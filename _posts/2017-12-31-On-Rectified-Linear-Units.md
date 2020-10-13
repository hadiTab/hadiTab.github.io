---
layout: post
title:  "On Rectified Linear Units"
date:   2017-12-31 -0600
categories: machine_learning
mathjax: true
---
{% include mathjax.html %}

Rectified Linear Units (ReLU) are becoming the default method for adding nonlinearity to neuron activations for most layers in deep neural networks, largely due to the success of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

ReLU are saturating non-linearities which are able to significantly reduce the training time compared to equivalent DNNs using tanh or sigmoid activations, using gradient descent. Rectifier units negate the negative part of a neurons activation, keeping the positive portion:

$$ f(x) = x^+ = \max(0, x) $$

ReLU keeps the positive portion of the gradient, which results in reduced likelyhood of the gradient vanshing (a major problem with sigmoid and tanh activation functions). The negative porition, on the other hand, is negated which results in a sparse representation.

The zero-thresholding can create some problems as well. ReLU neurons are prone to getting stuck at zero when a large negative gradient passes through. This can result in the weight never getting adjusted anymore as the neuron will not be activated. This is referred to as the *dying ReLU* problem. Proper initialization can reduce the possibility of having a large precentage of dead ReLU units.

Several remedies and alternatives have been proposed to address the issues with ReLU.

### Leaky ReLU

The leaky ReLU has a slight negative slope to counteract the complete negation of negative gradients. The slope is set at a constant value, typically 0.01.

$$ f(x)=\begin{cases} 
x \quad \quad \quad x > 0\\ 
0.01x \quad \text{otherwise} 
\end{cases} $$

This helps avoid dying ReLU units. It is worth noting that the representation will no longer be sparse if leaky ReLU is used. The leaky ReLU has been shown to have a negligable effect on model accuracy.

### PReLU

The parametric rectified linear unit (PReLU), proposed by [He et al.](https://arxiv.org/pdf/1502.01852.pdf) at Microsoft Research. The PReLU takes the leaky ReLU one step further by turning the slope of the negative portion into a trainable parameter.

$$
f(x)=\begin{cases} 
x \quad \quad \quad x > 0 \\ 
ax \quad \text{otherwise} 
\end{cases} 
$$

Where $a$ controls the slope of the negative part. He et al. also propose initializing the biases to zero and the weights to be randomly drawn from a zero-mean Gaussian distribution with a standard deviation of $\sqrt{2/n_l}$, where $n_l$ is the number of units in the layer.


### cReLU

[Concatenated Rectified Linear Units (cReLU)](https://arxiv.org/pdf/1603.05201.pdf) enforces non-saturated nonlinearity and preserves information from both the positive and negative portions. The idea for cReLU came about when the authors discovered that the learned filters in the early layers of deep CNNs such as AlexNet appear to form pairs of polar opposites within the a single conv layer. cReLU looks to enhance training by removing this redundancy.
cReLU simply concatenates the linear response of the ReLU activation on $x$ and $-x$.

### ELU

The [Exponential Linear Unit (ELU)](https://arxiv.org/pdf/1511.07289.pdf) scales the negative sideusing an exponential function:

$$
f(x)=\begin{cases} 
x \quad \quad \quad \quad x > 0\\ 
\alpha(e^x – 1) \quad \text{otherwise} 
\end{cases} 
$$

Where $\alpha$ is a tunable hyperparamater, often set equal to 1, which sets the degree of saturation for negative inputs. The effect of the exponential functoin on the negative side is that it brings the mean activation closer to zero and reduces bias shift, an effect similar to *batch normalization*. The ELU has been shown to speed up convergence and improve accuracy on CIFAR-100. ELU is intended to outperform models that combine ReLU and batch-normalization

### PELU

The [Parametric Exponential Linear Unit (PELU)](https://arxiv.org/pdf/1605.09332.pdf) is a parameterization of both the positive and negative side of ELU with learnable parameters $a$ and $b$.

$$
f(x)=\begin{cases} 
\frac{a}{b}x \quad \quad \quad \quad x \geq 0\\ 
a(e^{(\frac{x}{b})} – 1) \quad \text{otherwise} 
\end{cases} 
$$

Compared to ELU, PELU seems more effective at keeping the activation mean near zero in deeper networks.
The results indicate an improved performance over ELU and effectiveness in minimizing vanishing gradients and bias shift.

### SELU

Scaled Exponential Linear Units (SELU), first proposed in a paper titles [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf) (SNN) are a type of activation function similar to ELU which strives to keep the activation between 0 and 1. Neuron activations of SNNs automatically converge toward zero mean and unit variance. An upper and lower bound is imposed on activation variance which theoretically makes vanishing gradients impossible.

$$
f(x)=\lambda\begin{cases} 
x \quad \quad \quad \quad x > 0\\ 
\alpha(e^{x} – 1) \quad \text{otherwise} 
\end{cases} 
$$

A SELU is essentially an ELU multiplied by $\lambda > 1$. This allows scaling of the activation for both negative and positive cases. For initialization, a random selection from a standard normal distribution is proposed.

Although not explicitly mentioned in the paper, from the [source code](https://github.com/bioinf-jku/SNNs) it seems that the values for the scaling hyper-parameters are:

$$
\alpha = 1.6732632423543772848170429916717 \\ 
\lambda = 1.0507009873554804934193349852946 
$$

### Swish

In an effort to find the ‘best’ activation function, the Google Brain team used an automated search method based on reinforcement learning to <em>learn</em> the best activation function. The results were published [here](https://arxiv.org/pdf/1710.05941.pdf). The best discovered function has been named swish and is:

$$
f(x) = x.\text{sigmoid}(\beta x)
$$

The team found the function to work better than ReLU across a number of challenging network.
It seems that the idea of $x.\text{sigmoid}$ was introduced earlier in a [paper](https://openreview.net/pdf?id=Bk0MRI5lg) by Hendrycks et al. and they concluded that $x.\text{CDF}$ performed better for a Gaussian distribution.