# Investigating the working memory capacity of recurrent neural network models

## Background

to hold information in working memory or integrate sensory inputs over time, the brain must be able to maintain representations of past sensory inputs for several seconds for future computation (NMA: [W3D2](https://compneuro.neuromatch.io/tutorials/W3D2_HiddenDynamics/chapter_title.html); [Barak and Tsodyks, 2014](https://doi.org/10.1016/j.conb.2013.10.008)). recurrently connected networks of neurons have been shown to be capable of producing stimulus-evoked firing rate fluctuations on timescales many orders of magnitude longer than that of the membrane time constant of individual neurons ([Chaudhuri and Abbott, 2011](https://doi.org/10.1523/JNEUROSCI.1677-11.2011)). in this project, we ask: what strategies might recurrently connected populations in the brain employ to preserve temporally structured information over time?

## Project setup

[link to Colab notebook](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/neurons/working_memory_capacity_rnn.ipynb)

## Project map

the project core is covered by Q1–5; subsequent questions can be taken in any order.

## Questions

### Q1

train a decoder to predict the input identity at time (t-X) from the firing rate of the N neurons at time t, for various positive values of X. you should use a separate decoder for each X. plot decoder performance as a function of X, for g=1.

### Q2

now, try varying the value of g: how does decoder performance change? generate the plots from question 1 for several values of g.

### Q3

next, try tuning the input magnitude, or the number of cells in the network to receive input. generate plots of decoder performance for several values of these two parameters.

### Q4

rather than a single input pulse, consider a network receiving a stream of inputs over time. how do new inputs affect decoder performance for past inputs?

### Q5

consider the input to your network. different input values can be passed as pulses of different magnitudes, or as inputs to different subsets of neurons. which input form can persist for longer?

### Q6

we initially assumed gaussian random weights in our network. how does memory capacity change if we enforce Dale's law?

### Q7

a linear decoder can only tell us so much. to incorporate nonlinear effects, compute the mutual information between network firing rates and the input identity, as a function of time, for multiple values of g.

### Q8

what kinds of solutions do we get if we train network weights to maximize memory instead, eg using FORCE? how does memory capacity change?

### Q9

test unsupervised plasticity mechanisms such as Hebbian/anti-Hebbian/asymmetric synaptic plasticity. how does memory capacity change?

## Author

made by Ann Kennedy
Northwestern University
ann.kennedy@northwestern.edu
