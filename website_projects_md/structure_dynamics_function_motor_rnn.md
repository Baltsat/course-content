# Linking structure, dynamics and function in recurrent neural networks (RNNs) during motor learning

## Background

RNNs with diverse architectures ([Jaeger & Haas 2004](https://doi.org/10.1126/science.1091277); [Sussillo & Abbott 2009](https://doi.org/10.1016/j.neuron.2009.07.018)) can be trained to replicate a variety of target signals ([Sussillo et al. 2009](https://doi.org/10.1016/j.neuron.2009.07.018)), including primate's motor trajectories in reaching tasks ([Rajan et al. 2010](https://doi.org/10.1103/PhysRevE.82.011903); [Vyas et al. 2020](https://doi.org/10.1146/annurev-neuro-092619-094115)). Using a simple, mathematically convenient "linear RNN," ([Rajan et al. 2010](https://doi.org/10.1103/PhysRevE.82.011903)) showed how changes in RNN activity manifold and prior learning feedbacks predict the speed and performance of motor adaptation. We propose to explore how RNN representation dynamics is linked with its connectivity structure and the complexity of the task learnt ([Gao 2017](https://doi.org/10.1101/214262)).

## Project setup

We provide code and useful functions: [Colab](https://colab.research.google.com/), [cleaned-up GitHub](https://github.com/), [Original GitHub](https://github.com/).

**Project map:** The project relates to the courses on Dynamical System (Dynamic Networks).

---

## Questions

### Q1

Use [Colab](https://colab.research.google.com/) to train [Rajan et al. 2010](https://doi.org/10.1103/PhysRevE.82.011903) "motor reaching" RNN over 50 trials. Plot the time course of its loss function. Does loss decrease monotonically? If not, at what trial? Why not? Does learning converge? If yes, how fast is convergence?

### Q2

Instantiate RNNs with increasing connection densities, calculate the mean squared error (MSE, the reaching performance) between target and cursor velocities for each, after learning. Plot MSE against connection density?

### Q3

Can we maximize connectivity sparsity during training, while maintaining performance?

### Q4

Sparse networks are energy efficient. Calculate each RNN's path length, weighted path length, modularity. How fast is information propagation in dense vs. sparse networks? Does RNN structure evolve throughout learning?

### Q5

Design a simpler task, in which the network's target is a "force" signal oscillating at cued periodicity (a sinewave). Train a new instance of [Rajan et al. 2010](https://doi.org/10.1103/PhysRevE.82.011903)'s RNN to perform the task. Is 50 trials sufficient for convergence? Compare the dimensionalities of the neural manifolds of the reaching and force tasks after learning.

### Q6

Linear units are analytically convenient but biological neuron responses saturate. Use the tanh activation function. How does it affect the network's initial activity dynamics? Is it chaotic? Does it improve learning?

### Q7

Does RNN's learning and neural manifold change with task complexity? Change the number of target cues (motor reaching task) and periodicities (force task). Compare learning speeds. Does neural mode distribution change? How? Can you link manifold dimensionality with task complexity?

### Q8

Motor cortical circuits are capable of switching between tasks of varying complexity. How would you modify the network to enable learning and execution of the two tasks with minimal or no cross-talk?

### Q9

Can you design a task in which a single cue takes on one of two meanings, depending on context: 1) perform "motor reaching" or 2) "exert force". How would you modify the network to achieve this?

---

## References

- Jaeger, H. & Haas, H. (2004). Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication. *Science*, 304(5667), 78-80. DOI: [10.1126/science.1091277](https://doi.org/10.1126/science.1091277)
- Sussillo, D. & Abbott, L.F. (2009). Generating Coherent Patterns of Activity from Chaotic Neural Networks. *Neuron*, 63(4), 544-557. DOI: [10.1016/j.neuron.2009.07.018](https://doi.org/10.1016/j.neuron.2009.07.018)
- Rajan, K., Abbott, L.F. & Sompolinsky, H. (2010). Stimulus-dependent suppression of chaos in recurrent neural networks. *Physical Review E*, 82, 011903. DOI: [10.1103/PhysRevE.82.011903](https://doi.org/10.1103/PhysRevE.82.011903)
- Gao, P. et al. (2017). A theory of multineuronal dimensionality, dynamics and measurement. *bioRxiv*. DOI: [10.1101/214262](https://doi.org/10.1101/214262)
- Vyas, S., Golub, M.D., Sussillo, D. & Shenoy, K.V. (2020). Computation Through Neural Population Dynamics. *Annual Review of Neuroscience*, 43, 249-275. DOI: [10.1146/annurev-neuro-092619-094115](https://doi.org/10.1146/annurev-neuro-092619-094115)

---

## Author

Made by Steeve Laquitaine
EPFL, Blue Brain Project & KU Leuven
steeve.laquitaine@epfl.ch
