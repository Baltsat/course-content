# Supervised and unsupervised learning in recordings of 80,000 neurons

**Background:** Humans and other animals can learn to distinguish new visual features and categories after instruction (perceptual learning). This type of learning is thought to be mediated by enhanced neural responses to the learned stimuli in the visual cortex ([Poort et al., 2015](https://www.sciencedirect.com/science/article/pii/S0896627315004766)). While visual cortex responses indeed change during task learning, most of these changes can also happen without a task ([Zhong et al., 2024](https://www.nature.com/articles/s41586-025-09180-y)), similar to unsupervised learning in artificial neural networks. In this case, visual experience alone drives neural plasticity that may be useful for subsequent supervised learning. How supervised and unsupervised learning interact in the visual cortex remains poorly understood. The dataset from [Zhong et al., 2025](https://www.nature.com/articles/s41586-025-09180-y) includes large-scale 2-photon calcium imaging data (>50,000 neuron per recording) from multiple visual cortical regions, before and after learning (in different neurons), and after introducing novel visual stimuli.

**Project setup:** The colab notebook shows how to load and analyze individual sessions ([link](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/neurons/visual_learning_80k_neurons.ipynb)). You can also access the full dataset from [Figshare](https://janelia.figshare.com/articles/dataset/Unsupervised_pretraining_in_biological_neural_network/28811129/2) ([code](https://github.com/MouseLand/zhong-et-al-2025)), but this may require additional compute resources beyond those available in colab.

**Project map:** The project core is covered by Q1-4; subsequent questions offer multiple possible paths forward, or you can explore the dataset directly in colab. This [review paper](https://www.science.org/doi/full/10.1126/science.adp7429) has additional analysis ideas for large-scale neural data.

---

## Questions

### Q1
Since the data is very large (and does not fit easily in google colab), we have provided PCA-compressed neural responses. How do these relate to the original data? Can you do similar analyses on PCA-ed data as on raw data?

### Q2
Do neurons have position selectivity? How/why would that arise in visual cortex? Plot tuning curves, either from raw data or from PCA-ed data.

### Q3
What kinds of neural plasticity are driven by learning? Compare neural response selectivity before vs after task learning.

### Q4
What kinds of neural plasticity require task learning? What kinds of plasticity arise just from unsupervised exploration of the VR? Analyze the unsupervised learning sessions.

### Q5
If the quality of PCA allows for many statistical analyses to be performed in PCA space, and then transformed back to neural space. Can you show for several types of analyses? What are the advantages?

### Q6
Why is neural plasticity in visual cortex useful? Make a theory for how primary vs higher-order visual processing may need different types of plasticity for different purposes.

### Q7
Novelty responses are interpreted as failures of prediction based on a learned internal model of the world. Does this "predictive coding" theory make a prediction about novelty responses to "fast2" and "circles"?

### Q8
Visual processing relies on invariant object recognition. Can you find neurons that encode the textures in an invariant way? These neurons should respond everywhere in one corridor, but not in the other. Are these neurons "learned"?

### Q9
Instead of PCA, more complicated linear dimensionality methods can be used (ICA, NMF, GPFA). Try some. What are the advantages, if any, over PCA?

### Q10
Can you test your theory (from Q6, Q7 or Q11) in this dataset? If not, can you design a new experiment that would let you test the theory? Make explicit predictions, and also propose an alternative / null hypothesis for that.

### Q11
Invariant object recognition is typically described in hierarchical neural networks. Can you make a theory for how the brain regions described here (V1, medial, anterior, lateral) process visual information sequentially or in parallel to achieve invariant object recognition?

---

*Made by Lin Zhong and Marius Pachitariu*
*IBL Janelia*
*zhong.pachitariu@janelia.org*
