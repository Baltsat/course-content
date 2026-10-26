# Investigating the Visual Information in Different Brain Regions

## Background

Stimulus-evoked activity can be measured with fMRI through haemodynamics: changes in blood flow, volume, and oxygenation in response to neuronal activity changes. fMRI recordings allows recording responses from multiple brain regions simultaneously with high spatial resolution. In this project we look at the human fMRI response to visual stimulation and investigate which brain regions encode which type of visual information by predicting fMRI responses from artificial vision models.

## Project Setup

Consider fMRI response to visual stimulation (images/videos) and a basic computer vision model (AlexNet) performing image classification. For images you can use [Kay & Gallant dataloader](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_kay_images.ipynb), for videos you can use [Algonauts databaser](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_algonauts_videos.ipynb).

## Project Map

The project core is covered by Q1-3; subsequent questions can be taken in any order.

## Questions

### Core Questions (Q1-Q3)

**Q1.** Train a voxel-wise encoding model ([Yamins et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3037423/), [Cichy et al.](https://arxiv.org/ftp/arxiv/papers/2104/2104.13714.pdf)) to predict fMRI responses in different brain regions from different layers of Alexnet ([Krizhevsky et al.](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)).

**Q2.** Is visual cortex hierarchy related to model's hierarchy? Compare the predictions of individual layers for different regions by visualizing the results through a heatmap. Do you observe any pattern? (e.g. in [Guclu et al.](https://www.jneurosci.org/content/jneuro/35/27/10005.full.pdf), [Agrawal et al.](https://arxiv.org/pdf/1407.5104.pdf))

**Q3.** Now try predicting fMRI response using a randomly initialized AlexNet model. Is the pattern you observed different from above? What role does training the model on image classification have on prediction performance?

### Extended Questions (Q4-Q8)

**Q4.** Which brain regions preserve spatial information? Apply image transformations randomly before feeding to DNN and observe the change in encoding accuracy.

**Q5.** Which brain regions encode categorical information? The deeper layers of the model (fc6-fc8) are known to encode categorical information.

**Q6.** Which brain regions encode action information? Find a model performing action recognition and compare encoding accuracy of different regions predicted by deeper layers.

**Q7.** Find which DNN units play a major role in encoding performance. Remove DNN units and investigate the impact on encoding performance.

**Q8.** Visualize using [lucid/lucent](https://github.com/greentfrapp/lucent) or [network dissection](http://netdissect.csail.mit.edu/) what information is encoded in top units.

## Key References

- Krizhevsky A, Sutskever I, Hinton GE (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS. DOI: [10.1145/3065386](https://doi.org/10.1145/3065386)
- Guclu U, van Gerven MAJ (2015). Deep Neural Networks Reveal a Gradient in the Complexity of Neural Representations across the Ventral Stream. J Neurosci 35(27):10005-10014. DOI: [10.1523/JNEUROSCI.5023-14.2015](https://doi.org/10.1523/JNEUROSCI.5023-14.2015)
- Cichy RM, et al. (2021). The Algonauts Project 2021 Challenge: How the Human Brain Makes Sense of a World in Motion. arXiv:2104.13714. DOI: [10.48550/arXiv.2104.13714](https://doi.org/10.48550/arXiv.2104.13714)
- Bau D, et al. (2017). Network Dissection: Quantifying Interpretability of Deep Visual Representations. CVPR. DOI: [10.1109/cvpr.2017.354](https://doi.org/10.1109/cvpr.2017.354)

## Author

Created by Kshitij Dwivedi
Goethe University Frankfurt
dwivedi@em.uni-frankfurt.de
