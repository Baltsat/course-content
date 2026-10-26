# Mining naturalistic movement from human ECoG

## Background

The neural correlates of unstructured, spontaneous movements in completely naturalistic settings have rarely been studied, due in large part to a lack of available data. Here, you can use the Annotated Joints in Long-term Electrocorticography (AJILE12) from human participants (recorded during epilepsy monitoring). AJILE12 synchronized intracranial neural recordings and upper body pose trajectories across 55 semi-continuous days of naturalistic movements, along with relevant metadata, including thousands of wrist movement events and annotated behavioral states.

## Project setup

AJILE12 data is available on [DANDI](https://www.dandiarchive.org/). Use this repository [GitHub](https://github.com/neurovium/Neuromatch-AJILE12) for background and understanding the data. Notebook can run on [Colab Project template](https://github.com/neurovium/Neuromatch-AJILE12/blob/master/Notebook/exploreAJILE12.ipynb).

**Class 1** (green) is necessary, **Class 2** (yellow) is voluntary, and **Class 3** (red) is for future explorations if it interests you.

---

## Class 1 problems: A grand challenge in the exploration of methods that can be used to predict spontaneous activity and intentions

In the paper [Singh et al. (2021)](https://doi.org/10.1016/j.jneumeth.2021.109199), Singh et al used a "Random Forest" algorithm to discriminate between presence/absence of movement for each wrist using only features derived from the ECoG.

### Q1
Your first step is to be able to understand how to use RF as a decoder: Check the paper and understand the general concept. Hints: You can use "RandomForestClassifier" from "sklearn.ensemble" for implementing the RF. Here is a general outline of what you want to do:
- Split data into training and testing
- Define a RF classifier using RandomForestClassifier (you can start with what Singh et al used in their approach)
- Train the model using the training set
- Evaluate the model using the testing set

### Q1.1
Extract other features from the ECoG data and set your model to separate between meta-states of behavior (such as talking versus moving). Which features would you choose? Hints: Depending on your behaviors of choice, you may want to think about the electrode locations. Can you add spatial information to your classifier?

### Q2
Coarse-grained event metadata includes events such as reach angle, reach goal etc. Can you use RF and enriched features such as rhythm interactions or phase coupling to build your classifier?

---

## Class 2 problems: Classification problem, bias and overfitting are common in ML

Bias and overfitting are common in ML problems when you encounter a rare RF exception. To set a RF, you need to set number of features. You want enough trees to get the best predictive accuracy, but also you do not want so many trees that would take too long to train the model or push your system to run out of memory.

### Q3
How do you reduce overfitting and bias? Think about regularization techniques such as L1/L2.

### Q4
You can use a different method instead of RF to overcome some of the issues. For example, eXtreme Gradient Boosting (XGBoost) can be a good alternative (from xgboost import XGBClassifier).

---

## Class 3 problems: A problem with long recordings is the non-stationary nature of the data

### Q5
How do you deal with the non-stationarity of the signal over long recording time spans? Hint: To overcome the nonstationary nature of your data, you can use techniques such as normalization or differencing to remove non-stationary effects and fall back to the stationary case. Also, you can use techniques such as normalization or differencing to remove non-stationary effects and fall back to the stationary case. Try to add these to your classifier.

### Q6
Can you leverage unsupervised latent factor modeling to define new classes for the data? Hint: Independent component analysis (ICA) can be used to extract features from multimodal data. ICA is a blind source separation technique that can separate a multivariate signal into independent components. To extract latent factors from the results of ICA, you can use clustering techniques such as k-means clustering or hierarchical clustering to group similar components together. To link the latent factors to labeled metadata such as movement, sleep, talking etc., you can use supervised learning techniques such as support vector machines (SVM) or random forests to learn a mapping between the latent factors and the metadata labels.

---

## Credits

Designed by **Nima Dehghani**
[nima.dehghani@mit.edu](mailto:nima.dehghani@mit.edu)

---

## References

- Singh, S. H., Peterson, S. M., Rao, R. P. N., & Brunton, B. W. (2021). Mining naturalistic human behaviors in long-term video and neural recordings. *Journal of Neuroscience Methods*, 358, 109199. [https://doi.org/10.1016/j.jneumeth.2021.109199](https://doi.org/10.1016/j.jneumeth.2021.109199)
- Peterson, S. M., et al. (2022). AJILE12: Long-term naturalistic human intracranial neural recordings and pose. *Scientific Data*, 9, 184. [https://doi.org/10.1038/s41597-022-01280-y](https://doi.org/10.1038/s41597-022-01280-y)

## Resources

- [DANDI Archive](https://www.dandiarchive.org/) - Data repository
- [GitHub Repository](https://github.com/neurovium/Neuromatch-AJILE12) - Code and documentation
- [Colab Notebook](https://github.com/neurovium/Neuromatch-AJILE12/blob/master/Notebook/exploreAJILE12.ipynb) - Interactive exploration
