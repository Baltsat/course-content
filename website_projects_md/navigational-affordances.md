# Computational Modeling of Navigational Affordances and Scene Selective Cortical Responses

## Background

As we move about the world, we use vision to determine where we can go in our local environment and where our path is blocked. In this project, our first goal is to identify which brain regions encode the structure of navigable space in visual scenes using behavioral and fMRI data from [Bonner and Epstein](https://www.pnas.org/content/114/18/4793). The next goal is to identify a computational model of navigational affordances and underlying cortical responses.

## Project Setup

Consider fMRI response to indoor images in the form of ROI specific and searchlight representational dissimilarity matrices (RDMs). Find which ROIs/searchlights show significant correlation with navigational affordances behavioral RDM. Identify relevant computer vision models to explain navigational affordance behavioral and scene selective fMRI responses.

## Project Map

The project core is covered by Q1-4; subsequent questions explore potential models to explain behavioral and scene-selective fMRI responses.

## Questions

### Core Questions (Q1-Q4)

**Q1.** Compare behavioral navigational affordance RDMs with ROI RDMs using Spearman's correlation. Which ROI shows highest correlation with affordance RDMs?

**Q2.** Compare behavioral navigational affordance RDMs with subject-wise searchlight RDMs.

**Q3.** Perform a t-test to find out which searchlights show significant correlation with navigational affordance RDM.

**Q4.** Input the stimuli to a computer vision model (e.g. AlexNet) and create layer RDMs. Correlate the model RDMs ([Bonner & Epstein 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006111)) with fMRI and behavioral RDMs. Which layer shows the highest correlation? Why?

### Extended Questions (Q5-Q7)

**Q5.** Think of other relevant computer vision tasks that are more related to navigational affordances (e.g. [Dwivedi et al.](https://direct.mit.edu/jocn/article/doi/10.1162/jocn_a_01624/97376/Unraveling-Representations-in-Scene-selective), [Lescroart et al.](https://www.cell.com/neuron/pdf/S0896-6273(18)30995-4.pdf)). Perform the same analysis as previous steps and compare results.

**Q6.** Compare the two model RDMs. Do you observe a high correlation between them? Apply variance partitioning analysis to find how much of the variance of fMRI and behavioral RDMs these models explain uniquely.

**Q7.** Based on the functions of scene selective regions ([Lescroart et al.](https://www.cell.com/neuron/pdf/S0896-6273(18)30995-4.pdf), [Stansbury et al.](https://www.cell.com/neuron/pdf/S0896-6273(19)30349-6.pdf)) identify computer vision models that perform a similar function. Correlate with fMRI ROI RDMs and perform permutation test to validate your hypothesis.

## Key References

- Bonner MF, Epstein RA (2017). Coding of navigational affordances in the human visual system. PNAS 114(18):4793-4798. DOI: [10.1073/pnas.1618228114](https://doi.org/10.1073/pnas.1618228114)
- Bonner MF, Epstein RA (2018). Computational mechanisms underlying cortical responses to the affordance properties of visual scenes. PLoS Comput Biol 14(4):e1006111. DOI: [10.1371/journal.pcbi.1006111](https://doi.org/10.1371/journal.pcbi.1006111)
- Dwivedi K, et al. (2021). Unraveling Representations in Scene-selective Brain Regions Using Scene Parsing Deep Neural Networks. J Cogn Neurosci 33(10):2032-2043. DOI: [10.1162/jocn_a_01624](https://doi.org/10.1162/jocn_a_01624)
- Lescroart MD, Gallant JL (2019). Human Scene-Selective Areas Represent 3D Configurations of Surfaces. Neuron 101(1):178-192. DOI: [10.1016/j.neuron.2018.12.014](https://doi.org/10.1016/j.neuron.2018.12.014)

## Author

Created by Kshitij Dwivedi
Goethe University Frankfurt
dwivedi@em.uni-frankfurt.de
