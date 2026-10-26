# Attractor models of short-term memory

## Background

Persistent, stimulus-dependent neuronal activity has been observed in some brain areas during tasks (that require) requiring the temporary maintenance of information. This function is called short-term memory. Neurons representing the information can remain active for several seconds after the original input has been removed. In this project you will build a simple, population-based, dynamical system model of a short-term memory task (NMA: [W2D4](https://compneuro.neuromatch.io/tutorials/W2D4_DynamicNetworks/chapter_title.html)). You will study how even a simple model can generate different behaviors. This project requires basic proficiency in the mathematics of dynamical systems.

## Project setup

[click to a google colab>](https://colab.research.google.com/)

## Project map

Q1-5 cover the core part of this project. Additional questions may emerge while tackling Q1-5. Explorations are encouraged.

---

## Questions

### Q1

Simulate a dynamical system consisting of two excitatory units and one inhibitory unit connected to both excitatory units. Use firing rate equations and a sigmoidal nonlinearity as the transfer function. Allow each unit to receive an external input.

### Q2

Calculate and plot the two nullclines of the system, assuming that the inhibitory unit membrane instantaneously integrates all inputs.

### Q3

Find a set of network parameters that allows the network to perform a binary task with short-term memory. Deliver a positive stimulus to either of the excitatory units to simulate the two trial types. What parameter(s) can you vary so that the network displays two fixed points? How do you push the network activity towards one of the fixed points?

### Q4

Verify the stability of the network by perturbing its activity with a transient stimulus. Can you make the network robust to perturbation of the inhibitory unit (mimicking an optogenetic perturbation experiment)? Can you make the activity switch between fixed points?

### Q5

The value of the membrane time constant typically lies within the 5-150 ms range. However, neuronal activity appears to evolve on a slower (>500 ms) time course in short-term memory tasks. Can you tune the weights so that the network activity evolves toward the stable fixed points on a timescale slower than the typical membrane time constant?

### Q6

A different proposed mechanism for short-term memory relies on line-attractors. Can you tune the network so that it produces a line attractor? Instead of a sigmoid, try using a rectified linear transfer function. Does this help?

### Q7

Short-term plasticity occurs when the availability of the neurotransmitters varies over time, causing a history-dependent increase or a decrease of the firing rate. Study the network after adding short-term facilitation (STF) and short-term depression (STD). How do STF and STD change the shape of the nullclines? How does the dynamic change? Does the activity at the fixed point still appear steady?

### Q8

Add a slowly ramping external input to the excitatory units. Find a set of parameter values that make the network switch from one to two stable fixed points (i.e. that will cause a bifurcation) as a function of the external input value.

---

## References

- [Dynamic Networks - Neuromatch Academy: Computational Neuroscience](https://compneuro.neuromatch.io/tutorials/W2D4_DynamicNetworks/chapter_title.html)
- [Dynamical systems, attractors, and neural circuits (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4930057/)
- [Stability of working memory in continuous attractor networks under the control of short-term plasticity (PLOS Comp Bio)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006928)

---

**Made by Lorenzo Fontolan**
Janelia Research Campus
fontolanlATjanelia.hhmi.org
