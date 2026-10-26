# Investigating state-dependent decision-making in mice performing a 2AFC task

**Background:** The International Brain Laboratory studies perceptual decision-making in mice. They have collected millions of trials of mice making a binary choice based on a visual stimulus. Formally, this is called a two alternative forced choice (2AFC) task. While the mice learn to perform the task well, they still make many mistakes. This project will study these "lapses" in judgement to look for evidence of time-varying internal states, following recent work of [Ashwood et al (2021)](https://doi.org/10.1101/2020.01.29.925610) and [Roy et al (2021)](https://doi.org/10.1038/s41593-021-00891-x).

**Project setup:** This project will use the IBL behavior dataset ([psychometrics](https://int-brain-lab.github.io/iblenv/notebooks_external/loading_trials_data.html)).

**Project map:** The core project follows Q1-Q5, then branches into more involved models and analyses.

---

## Questions

### Q1
Fit a logistic regression for each fully-trained mouse to predict its choices given the stimulus contrast and a bias. This yields a classic psychometric curve. How well does the curve fit the data? Evaluate goodness of fit by comparing predicted vs empirical choice probabilities as a function of stimulus contrast (ignoring "violation" trials). Quantify performance with log likelihood on a test set.

### Q2
How can you change the model to account for trained mice that still make mistakes, even on high contrast stimuli? Introduce a lapse probability into the logistic regression (eq. 1-4 of [Ashwood et al., 2021](https://doi.org/10.1101/2020.01.29.925610)). Fit with gradient descent, and evaluate the model. Fit, and compare its performance on held-out data.

### Q3
Are choices autocorrelated? Include past choice and reward into the logistic regression and reanalyze your model, comparing it to those from Q1 and Q2.

### Q4
Is mouse behavior stationary; i.e. do mice seem to follow a single strategy? Fit the model from Q3 on sliding windows of data within a session. Do the parameters change significantly within a single session?

### Q5
Are some mice more stationary than others? I.e. do some mice seem to use one strategy the whole time whereas others change their behavior throughout a session (as measured by the logistic regression parameters in each window)? Similarly, do late show differences in stationarity of behavior?

### Q6
Does behavior change with the number of rewards received? For example, maybe mice become sated or "bored" over the course of the session. Do you see any evidence for such changes in internal state?

### Q7
Perhaps there is a "mental cost" to task engagement. Can you build a theoretical model to predict and explain changing behavior as rewards are accrued over a session?

### Q8
Following [Ashwood et al (2021)](https://doi.org/10.1101/2020.01.29.925610), fit a GLM-HMM for discrete internal states with different logistic regression parameters. Use EM to infer the states and estimate the model parameters. Hint: you might want to start with a library like [ssm](https://github.com/lindermanlab/ssm).

### Q9
The HMM from Q8 assumes discrete changes in internal state, but satiety and engagement might be described as continuous latent variables. Following [Roy et al (2021)](https://doi.org/10.1038/s41593-021-00891-x), build a GLM with continuously varying weights and fit it to the behavior data.

---

*Made by Scott Linderman*
*Stanford University*
*scott.linderman@stanford.edu*
