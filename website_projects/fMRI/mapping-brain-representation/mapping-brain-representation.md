# Mapping Brain Activation and Representation with fMRI

## Background

Stimulus-evoked activity can be measured with fMRI through haemodynamics: changes in blood flow, volume, and oxygenation in response to neuronal activity changes.

## Project Setup

Here we start from time series data. Using the HCP task data, where different stimuli are either ON or OFF, we will create increasingly sophisticated models of brain activity. We will then use these models to ask statistical questions on localisation of brain function and on information representation.

## Project Map

The project core is covered by Q1-4; subsequent questions can be taken in any order. If using data sets that have beta maps, you can start from Q5-6.

## Questions

### Core Questions (Q1-Q4)

**Q1.** Calculate the average activity levels during presentation of the stimulus and compare to when the stimulus is off. Contrast different brain areas. Generate summary plots etc.

**Q2.** Use a T-Test to formally test which regions' activity is significantly modulated by the stimulus, or significantly more for one stimulus type than another.

**Q3.** Set up a GLM where each regressor is a stimulus response model (e.g. ON-OFF) and regress this model against the fMRI time series data. Calculate the betas from this GLM and uncertainty on the betas. Use these to setup a statistical test for asking which regions respond to the stimulus.

**Q4.** Augment the GLM with a more realistic response profile by convolving the On-Off stimulus vectors with a haemodynamic response function (e.g. a Gaussian or Gamma function).

### Extended Questions (Q5-Q8)

**Q5.** Use the estimated beta maps to calculate RDMs (representation dissimilarity matrices) and map their variations across the brain.

**Q6.** Set up a linear decoder that can predict the stimulus type from the multivariate brain activity, either whole brain or dimensionality-reduced.

**Q7.** Design a statistical test that allows you to compare brain representations to a predefined template.

**Q8.** Contrast the results to an artificial neural network that is designed to solve a similar task/problem.

## Author

Created by Saad Jbabdi
University of Oxford
saad@fmrib.ox.ac.uk
