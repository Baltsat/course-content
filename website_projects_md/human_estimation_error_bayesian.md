# Probing the dynamics of human estimation errors in a Bayesian framework

## Background

When stimuli are uncertain, humans can use a rich set of strategies to exploit environment statistics to improve perceptual decisions ([Kording & Wolpert 2004](https://doi.org/10.1038/nature02169); [Laquitaine & Gardner 2018](https://doi.org/10.1016/j.neuron.2017.12.011)). It has been shown that a suboptimal but computationally cheap switching heuristic approximation of Bayesian inference explains human motion direction estimation better than sophisticated Bayesian observers. But switching-related estimate errors might not be the only cause of errors in the task. We propose to characterize subject perceptual errors and their dynamics, and to assess whether an online Bayesian model can provide a better fit to the data.

## Project setup

We provide code and useful functions: [Colab: biased decisions](https://colab.research.google.com/), [Colab: prior learning](https://colab.research.google.com/), [paper's GitHub](https://github.com/).

**Project map:** The project relates to the courses on Stochastic processes (Bayesian decisions).

---

## Questions

### Q1

Calculate the circular distance between displayed motion direction and subject estimate (estimate error) at each trial. Plot the errors time course. Are errors stable over trial? If not why?

### Q2

Calculate the circular distances between past trial estimates and current trial displayed directions. Split the data into a train and test set. Train a linear regression to predict the error from that feature on the train set and calculate the fit's R-squared on the test set. How good is the fit? Can you find more predictive features?

### Q3

Long exposure to stimulus motion can produce a "waterfall" aftereffect illusion ([Stocker & Simoncelli 2006](https://doi.org/10.1038/nn1669)). Do subject errors indicate that they experience this illusion or "cardinal biases" ([Girshick et al. 2011](https://doi.org/10.1038/nn.2831))?

### Q4

Can you train a classifier that uses these features to predict when subjects will switch to a direction within 45 degrees of the most frequently displayed direction (the experimental prior)?

### Q5

What task variables drive subject learning and how? Train an "online" Bayesian model that learns from the feedback received at the end of each trial, to reduce estimation error over trials.

### Q6

Fit the model to a subject's estimates. Assuming that subjects learn quickly (within 10 trials), fit a "static" Bayesian observer to his estimates. Use Akaike's information criterion (AIC) to identify the model that fits the data best.

### Q7

Do subject overgeneralize? I.e., do they transfer the prior learnt in a block to the next? Does this "strategy" account for a large proportion of estimate errors?

### Q8

Simulate a Hierarchical Bayesian observer (read [Lee and Mumford 2003](https://doi.org/10.1364/josaa.20.001434)) that "believes" that in some trials the prior is uniform, and in others it is circular (e.g., a Von Mises). Can this model reproduce the observed bimodality of subject estimate distributions?

### Q9

Show how such a hierarchical model could be learnt by using a state prediction error signal (see [Colab: prior learning](https://colab.research.google.com/)). Can this model explain errors not accounted for by the "online" Bayesian model?

---

## References

- Kording, K. & Wolpert, D. (2004). Bayesian integration in sensorimotor learning. *Nature*, 427, 244-247. DOI: [10.1038/nature02169](https://doi.org/10.1038/nature02169)
- Laquitaine, S. & Gardner, J.L. (2018). A Switching Observer for Human Perceptual Estimation. *Neuron*, 97(2), 462-474.e6. DOI: [10.1016/j.neuron.2017.12.011](https://doi.org/10.1016/j.neuron.2017.12.011)
- Stocker, A.A. & Simoncelli, E.P. (2006). Noise characteristics and prior expectations in human visual speed perception. *Nature Neuroscience*, 9(4), 578-585. DOI: [10.1038/nn1669](https://doi.org/10.1038/nn1669)
- Girshick, A.R., Landy, M.S. & Simoncelli, E.P. (2011). Cardinal rules: visual orientation perception reflects knowledge of environmental statistics. *Nature Neuroscience*, 14(7), 926-932. DOI: [10.1038/nn.2831](https://doi.org/10.1038/nn.2831)
- Lee, T.S. & Mumford, D. (2003). Hierarchical Bayesian inference in the visual cortex. *Journal of the Optical Society of America A*, 20(7), 1434-1448. DOI: [10.1364/josaa.20.001434](https://doi.org/10.1364/josaa.20.001434)
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19, 716-723.

---

## Author

Made by Steeve Laquitaine
EPFL, Blue Brain Project, KU Leuven
steeve.laquitaine@epfl.ch
