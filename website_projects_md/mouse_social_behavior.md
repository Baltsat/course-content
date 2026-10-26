# Analyzing the structure of mouse social behavior

## Background

Animals engaged in social behavior make continuous decisions of how to interact with a member of their species. In the field of ethology, we study the structure of animal behavior to gain insight into the neural algorithms by which that behavior is controlled (see e.g., [Anderson and Perona 2014](https://doi.org/10.1016/j.neuron.2014.09.005); WSDB). In this project, you'll explore mouse behavior in a resident-intruder assay, in which a male mouse in its home cage is presented with a male or female intruder, and the animals are allowed to freely interact. Can you determine from tracking data what cues shape animals' behavior? What timescales of behavior can you identify in the data?

## Project setup

Use the [CalMS21 dataset](https://arxiv.org/abs/2104.02710) ([Sun et al. 2021](https://arxiv.org/abs/2104.02710)) to study the structure of mouse social behavior. The poses of both animals have been tracked using a computer vision pipeline ([Segalin et al. 2020](https://doi.org/10.7554/eLife.63720)), and three social behaviors of interest (attack, mount, and sniffing) are manually annotated. Code to load and explore the dataset is provided [here](https://colab.research.google.com/drive/1PhcKzLgdEWJSK-HDTaD82fJdSSs5Sq_p?usp=sharing).

## Project map

An example project exploring timescales of behavior is outlined below – however feel free to get creative and follow other lines of investigation!

## Questions

### Q1
Visualize behavior matters for several example sequences, and plot likelihood of observing attack, mounting, and sniffing as a function of time since sequence start, for all sequences > 9 minutes long. Does behavior look stationary?

### Q2
Create and visualize some hand-crafted features characterizing animal pose – such as velocity, joint angles, inter-animal distance, etc. (see [Segalin et al. 2020](https://doi.org/10.7554/eLife.63720) for more ideas.)

### Q3
Compute event-triggered-averages of some of your features, triggering on onset or termination of attack/mounting/sniffing. Do you see any features that seem modulated by behavior?

### Q4
Are all bouts annotated for a given behavior the same? Try clustering frames or bouts of a behavior like attack or sniffing into "subtypes" based on the values of the features you created (You may want to look at average feature values across the bout, feature value at bout onset, etc.) Can you identify classes of aggression/investigation/mounting? (You should omit "other" from this analysis.)

### Q5
Create a transition matrix characterizing transitions between behaviors, including the behavior subtypes you identified in Q4 (omit "other" for this analysis).

### Q6
Revisit Q4, but now see if you can identify meaningful "subtypes" of the "other" category: start by considering the pose of one mouse at a time, and omit "other" frames that are close in time (within ~2 seconds) to annotated frames for simplicity. Can you detect any interesting behaviors like grooming or rearing?

### Q7
How non-Markovian are mouse behaviors? Compute the long timescale transition matrix p(behavior k == x | behavior N-k == y) for k = 5, 10, ... as in [Figure 2 here](https://www.nature.com/articles/s41593-019-0533-x). How do these compare to the predicted matrices you'd get assuming the system is Markovian?

## References

- Anderson, D.J. & Perona, P. (2014). Toward a Science of Computational Ethology. *Neuron*, 84(1), 18-31. DOI: [10.1016/j.neuron.2014.09.005](https://doi.org/10.1016/j.neuron.2014.09.005)
- Sun, J.J. et al. (2021). The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions. *NeurIPS 2021 Track on Datasets and Benchmarks*. arXiv: [2104.02710](https://arxiv.org/abs/2104.02710)
- Segalin, C. et al. (2021). The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice. *eLife*, 10:e63720. DOI: [10.7554/eLife.63720](https://doi.org/10.7554/eLife.63720)

## Author

Made by Ann Kennedy
Northwestern University
ann.kennedy@northwestern.edu
