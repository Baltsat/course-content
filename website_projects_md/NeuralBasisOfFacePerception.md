# Does neural activity reflect face perception?

## Background

It is all too easy to find a neuron in the brain representing sensory features of the external world. This especially applies to visual cortex, where it has been shown that neurons in the mouse brain encode stimulus orientation to an average precision of 0.3 deg, which is 100x better than the mouse's behavioral performance (see [Stringer orientations dataset; Stringer et al., 2021](https://www.cell.com/cell/fulltext/S0092-8674(21)00373-1)). However, just because a population of neurons encodes a stimulus does not mean that the subject has conscious access to this information. When a subject is aware of a stimulus feature, we say that the subject has formed a "percept". Can we find areas of the brain where neural activity correlates with perception, measured psychometrically by the subject's performance in a task? In this project, you will answer this question for the Faces/Houses ECoG dataset ([Miller et al. 2017](https://journals.physiology.org/doi/full/10.1152/jn.00113.2017)), which contains stimuli with different noise levels as well as psychometric performance of the subjects viewing these stimuli.

## Project setup

Use the [data loader](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/ECoG/load_ECoG_faceshouses.ipynb) to investigate the ECoG responses.

---

## Project map

### Q1
Make a plot of the average responses of each electrode to faces and houses without noise (first experiment). Decide on the right time period to include for analyses, which distinguishes faces and houses.

### Q2
Formulate a selectivity index that tells you how well a given electrode distinguishes between faces and houses.

### Q3
On the noisy face data, make average responses for faces and house for subsets of noise fractions (i.e. 0 to 0.2, 0.2 to 0.4 etc). Compute your selectivity index again as a sanity check that the same electrodes are selective.

### Q4
Now compute the psychometric performance of the subject based on the fraction of true positives (correctly identified a face, TP) and false negatives (responded to houses thinking it was a face, FN).

### Q5
You now have all the ingredients to compare the visual responses to noisy images of faces, with the subject's performance for a given noise level.

### Q6
Can you predict a subject's mistakes on single-trials based on a single-channel of neural responses to the stimulus? If so, the neural activity might be directly reflecting the noisy perception.

### Q7
Make a model of each electrode's trial-to-trial variability based on the subject's responses. Compare this to a null model where each channel responds with its trial-averaged PSTH on every trial. If your model with the single-trial variability predicts more variance, does that mean the behavioral response influences the neural activity, or the other way around?

### Q8
Same as Q6 but now combining information across channels. You might need some kind of decoder for this; choose the right one from sklearn. Try to estimate how much the performance should improve if the information in each channel was independent from other channels.

### Q9
Different electrodes correlated in the way they predict behavioral mistakes? Try to predict...

### Q10
Now to answer the big question: does neural activity in the fusiform gyrus reflect face perception, or does it merely reflect the subject's motor response? Use the reaction time information to parse early vs late neural activity and repeat all analyses.

---

## Credits

Made by **Marius Pachitariu**
Janelia Research Campus
[pachitarium@hhmi.org](mailto:pachitarium@hhmi.org)

---

## References

- Miller, K. J., Hermes, D., Pestilli, F., Wig, G. S., & Ojemann, J. G. (2017). Face percept formation in human ventral temporal cortex. *Journal of Neurophysiology*, 118(5), 2614-2627. [https://doi.org/10.1152/jn.00113.2017](https://doi.org/10.1152/jn.00113.2017)
- Stringer, C., Michaelos, M., Tsyboulski, D., Lindo, S. E., & Pachitariu, M. (2021). High-precision coding in visual cortex. *Cell*, 184(10), 2767-2778. [https://doi.org/10.1016/j.cell.2021.03.042](https://doi.org/10.1016/j.cell.2021.03.042)

## Resources

- [Data Loader Notebook](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/ECoG/load_ECoG_faceshouses.ipynb) - Colab notebook to load the dataset
- [Miller Lab ECoG Data Library](https://searchworks.stanford.edu/view/zk881ps0522) - Stanford repository with full dataset
