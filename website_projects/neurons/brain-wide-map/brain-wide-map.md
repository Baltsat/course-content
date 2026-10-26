# Brain-wide map of neural activity during complex behaviour

**Background:** A key challenge in neuroscience is understanding how neurons in hundreds of interconnected brain regions integrate sensory inputs with prior expectations to initiate movements and thereby make decisions. Using a standardized behavioral task deployed across 12 labs and high density Neuropixels recordings, the International Brain Laboratory (IBL) has collected a brainwide-map dataset that enables the study of how distributed brain regions contribute to sensory-guided decisions. In this project, you will explore both single-neuron and population-level activity to uncover how task variables are encoded across the brain.

**Project setup:** To get started you can follow this [tutorial](https://colab.research.google.com/drive/1WEhmZF4hnxkuGCcqJn1tl5oGFgO85Qai#scrollTo=LSo26ge7nOBX).

**Project map:** The project core is covered in questions 1-5. More advanced analysis can then be explored in subsequent questions.

---

## Questions

### Q1
Explore the IBL brainwide map [visualisation website](https://viz.internationalbrainlab.org/app?spikesorting=ss_2024-05-06) and look for sessions that contain units that respond to the task stimulus. Load in the data and compare the firing rate before and after stimulus onset to detect responsive units.

### Q2
Extend this to all units in the different brain regions. Which regions are most responsive to the stimulus, does this make sense given the sensory input? Compare these results to those in [International Brain Lab et al.](https://doi.org/10.1101/2023.07.04.547681). Do they match?

### Q3
For responsive units compute the latency after stimulus onset of the response. Group the regions anatomically. Do the latencies make sense in terms of how sensory signals propagate through the brain?

### Q4
Are there units that vary their responses according to the block side? Is the visual stimulus encoded more robustly when it appears on the same side than the expected block prior? Does this effect vary with stimulus contrast?

### Q5
Repeat for choice. Which time window should we choose to remove movement confounds. Which areas encode choice?

### Q6
Look for responsive units that differentiate between stimulus side and stimulus contrasts. Are these only found in visual areas or can we find such units in other brain regions?

### Q7
From the neural activity predict the block side. Does stimulus contrast and timing within the block affect decoding performance?

### Q8
For areas that encode choice, explore how the activity of the units diverges between left and right choices prior to the choice being made. Which areas show a significant difference between left and right choices. At what latency before the movement is the choice made?

### Q9
From the neural activity predict whether the animal made a left or right choice. Use the trials data to assess the accuracy of the predictions.

### Q10
From the neural activity predict the stimulus side or the stimulus contrast. Use the trials data to assess the accuracy of the predictions.

---

*Made by Mayo Faulkner and Gaelle Chapuis*
*International Brain Laboratory*
*info@internationalbrainlab.org*
