# Retinotopic Mapping with fMRI

## Background

Multiple brain areas in the visual cortex contain a map of the visual/retinal space. fMRI is routinely used to estimate this mapping in order to delineate different subdivisions of the visual cortex and study their properties.

## Project Setup

This project is set to utilize a retinotopic mapping experiment, for which the Human Connectome Project data is used (see data [loader](https://colab.research.google.com/drive/1SgnnsZXx39k48MGi6D7mNmP-l7w-rc4v?authuser=1) and [Benson et al.](https://jov.arvojournals.org/article.aspx?articleid=2719988#207329261) for a detailed description of the data/experiment).

## Project Map

The project core is covered by Q1-3; For Q4 see [Dumoulin 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3073038/).

## Questions

### Core Questions (Q1-Q3)

**Q1.** Using a General Linear Model, calculate the magnitude and phase of the fMRI signal in every location of the brain.

**Q2.** Use the stimulus image files to adjust the phase of the stimulus. Use the borders of primary area V1 to validate.

**Q3.** Augment the model with convolution with a Haemodynamic response function to accommodate for the lag between neural and vascular responses.

### Extended Questions (Q4-Q5)

**Q4.** Fit a population receptive field (pRF) model to estimate the visual receptive field size in each brain area.

**Q5.** The previous Q involves nonlinear fitting, which can be time consuming over many brain voxels. Replace the nonlinear fitting with a deep ANN that can output pRF parameters directly from the data.

## Key References

- Dumoulin SO, Wandell BA (2008). Population receptive field estimates in human visual cortex. NeuroImage. DOI: [10.1016/j.neuroimage.2007.09.034](https://doi.org/10.1016/j.neuroimage.2007.09.034)
- Benson NC, et al. (2018). The Human Connectome Project 7 Tesla retinotopy dataset: Description and population receptive field analysis. Journal of Vision, 18(13):23. DOI: [10.1167/18.13.23](https://doi.org/10.1167/18.13.23)

## Author

Created by Saad Jbabdi
University of Oxford
saad@fmrib.ox.ac.uk
