# Effects of stimulus context and behavior state on visual representations

**Background:** Processing of visual information by cortical areas is influenced by animal's past experiences ([Poort et al., 2015](https://www.sciencedirect.com/science/article/pii/S0896627315004766); [Khan et al., 2018](https://www.sciencedirect.com/science/article/pii/S0959438818300400)) and modulated by the behavioral state of the animal ([Niell & Stryker, 2010](https://www.sciencedirect.com/science/article/pii/S0896627310000590)). While experience can lead to sparseness in neural encoding that persists over multiple days, behaviors like running, whisking, and overall animal's alertness can modulate activity in visual cortex on a second-to-second basis ([Stringer et al., 2019](https://www.science.org/doi/10.1126/science.aav7893); [Musall et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31551604/)). The cellular mechanisms that enable flexible neural coding over different timescales are still not well understood. The [Allen Institute's Visual Behavior 2P dataset](https://allensdk.readthedocs.io/en/latest/visual_behavior_optical_physiology.html) includes 2-photon calcium imaging data from excitatory and two subclasses of inhibitory neurons (SST and VIP) collected over six days under different experimental conditions: passive viewing vs active change detection task performance, with images that mice were trained on vs novel images they had no prior experience with.

**Project setup:** To get started, you can download a subset of data from only inhibitory cells ([example 1](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_Allen_Visual_Behavior_from_pre_processed_file.ipynb)) or access the full dataset using the [Allen SDK](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_Allen_Visual_Behavior_from_SDK.ipynb) ([example 2](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_Allen_Visual_Behavior_from_SDK.ipynb)).

**Project map:** Project ideas in green can be answered using pre-saved data file from example 1, while questions in yellow and red will require you to access data using AllenSDK, reformat it according to questions needs.

---

## Questions

### Q1
How do different subclasses of inhibitory neurons (VIP and SST) change their responses to visual stimuli when presented with images they have not seen before?

### Q2
In the dataset, 5% of image presentations were omitted at random. Do inhibitory neurons carry expectation signature when unexpected event happens, such as omitted stimulus?

### Q3
How do the changes in neural responses to stimuli or unexpected events vary by cortical area (V1, LM) or cortical depth? Add cortical area and depth as additional two dimensions in your analysis.

### Q4
Does activity of inhibitory neurons during unexpected events carry information about image identity? Train a decoder to predict image identity from neural activity during omitted stimuli.

How many different response profiles can be identified in the dataset? Do those profiles map onto cell classes?

### Q5
How does modulation of neural responses by behavioral state change with task engagement? Compare modulation by running within the same neurons in passive vs active sessions with familiar stimuli.

### Q6
Can neural changes seen with novel stimuli be explained by changes in animal behavior? Plot pupil and running traces around task related events for both familiar and novel sessions.

### Q7
Are different cell types modulated by animal behavior in a different way? Compare image responses of cell classes during stationary vs running trials.

### Q8
Model of cell class specific interactions: Is there a relationship in neural responses of three cell types that might reveal their unique functional role?

---

*Made by Iryna Yavorska*
*Allen Institute for Brain Science*
*iryna.yavorska@alleninstitute.org*
