# Detecting Affective States from Wearable Sensor Data

**CS-C3240 Machine Learning — Stage 1 Report**

---

## 1. Introduction

A modern smartwatch can report how many steps its wearer took and how long they slept, but not
whether they had a stressful day. The sensors needed for that are already on the device: heart
activity, skin conductance and body temperature all change measurably with emotional arousal. What
is missing is the step that turns those raw signals into a statement such as "the last hour was a
stressful one".

This project asks whether that step can be automated: **given a short window of physiological
signals recorded by a wearable device, can we predict which affective state the wearer was in?**
The motivation is adaptive user interfaces — a system that can read the user's state is a
prerequisite for one that responds to it, for example by suppressing notifications while the user
is concentrating.

Section 2 formalises this as a supervised classification problem. Section 3 describes the dataset,
feature construction, the chosen model and loss function, and the validation strategy. Section 4
reports the training and validation errors, and Section 5 concludes with the limitations of the
current approach.

---

## 2. Problem Formulation

**Data point.** One data point is a **60-second window** of physiological recordings from a single
subject, extracted with a 30-second stride. Windows overlapping a transition between two
experimental conditions are discarded so that each window carries an unambiguous label.

**Features.** Each data point is described by 21 continuous feature variables, computed as
summary statistics over the window (details in Section 3.2). They include heart-rate and
heart-rate-variability measures derived from the ECG signal, statistics of the electrodermal
activity (EDA) signal, respiration rate, body temperature, and accelerometer-based motion
statistics. All features are real-valued and continuous.

**Label.** The label is the experimental condition the subject was in during that window, a
categorical variable with three values: `baseline` (neutral), `stress` (a standardised social
stress test) and `amusement` (watching humorous video clips).

**Task type.** This is a **supervised learning** problem, specifically multi-class classification.
The aim is to learn a predictor that generalises to *subjects not seen during training*, since a
deployed system would have to work on a new user without per-person calibration.

---

## 3. Methods

### 3.1 Dataset

The data comes from **WESAD** (Wearable Stress and Affect Detection), a publicly available research
dataset published by Schmidt et al. [1] and hosted by the University of Siegen. It contains
recordings from 15 subjects wearing a chest-mounted RespiBAN device, which samples ECG,
electrodermal activity, respiration, body temperature and 3-axis acceleration at 700 Hz. Each
subject completed a study protocol of roughly 36 minutes with per-sample condition annotations.

Preprocessing: the meditation, recovery and undefined segments were removed; the remaining signal
was segmented into windows as described above, yielding **1024 data points**
(566 baseline, 311 stress, `[FILL]` amusement). The classes are imbalanced, as the
baseline condition is considerably longer than the amusement condition.

### 3.2 Feature selection

Features were chosen through two steps.

First, **domain knowledge**: the psychophysiology literature associates emotional arousal with
increased heart rate, reduced heart-rate variability and elevated skin conductance, so the initial
candidate set was built around signals known to carry that information rather than by taking all
available channels indiscriminately. Accelerometer features were added deliberately, because
physical movement produces artefacts in the autonomic signals; including motion statistics lets the
model account for movement instead of misattributing it to affect. The EMG channel was dropped, as
it mainly reflects local muscle activity and is not informative for the states of interest here.

Second, **inspection of the data**: A correlation heatmap of the candidate features revealed strong 
collinearity among several metrics. For instance, the correlation between mean, minimum, and maximum 
EDA was 1.00, and RMSSD strongly correlated with pNN50 (r = 0.96). Consequently, redundant features 
such as EDA min/max were removed. Furthermore, visualising the features via box plots grouped by condition 
showed that while accelerometer variance separated the stress class clearly, this likely reflects the 
physical movement required by the experimental stress protocol (standing and speaking) rather than the
 affective state itself. We also observed that body temperature trends might correspond more to the 
 elapsed time of the experiment rather than directly distinguishing the three emotional states.

One property we would have liked but could not obtain is the subject's own moment-to-moment
self-report of how they felt; the dataset provides only the experimental condition, which is a
proxy for the affective state rather than the state itself.

All features are standardised to zero mean and unit variance, with the scaling parameters computed
on the training split only and then applied unchanged to the validation split.

### 3.3 Model

The chosen model is **logistic regression**, whose hypothesis space consists of linear maps from
the feature vector to a vector of class scores, converted into class probabilities by the softmax
function.

This model was chosen for three reasons. It is the natural starting point for a classification
problem with continuous numeric features and no prior evidence that the decision boundary is
strongly non-linear. It has few hyperparameters, which matters given the modest number of data
points available. And its learned weights are directly interpretable: the coefficient on each
physiological feature can be compared against what the literature would predict, providing a
sanity check that the model is picking up a meaningful signal rather than an artefact. Establishing
how well a linear predictor performs also sets a baseline against which the more flexible models
considered in Stage 2 can be judged.

### 3.4 Loss function

The model is trained by minimising the **logistic loss** (multi-class cross-entropy) with L2
regularisation. The logistic loss is the natural choice here because it is the loss for which
logistic regression is defined, it is convex and therefore efficiently minimisable, and it is
sensitive to the *confidence* of predictions rather than only their correctness — which matters for
an application where a calibrated probability is more useful than a hard label.

For **evaluating** the learned hypothesis a different measure is used. Because the classes are
imbalanced, average 0/1 loss (i.e. accuracy) is misleading: a predictor that always outputs
`baseline` would already score deceptively well. The validation error is therefore reported as
**1 − macro-F1**, which averages the F1 score over the three classes with equal weight and so
penalises a model that ignores the smallest class. Accuracy is reported alongside it for
comparability with published results, together with the confusion matrix.

### 3.5 Construction of training and validation sets

The split is made **at the subject level, not at the window level**. Windows from the same subject
are highly correlated, and baseline physiology differs substantially between individuals. Under a
random split, windows from one subject would appear in both the training and validation sets,
allowing the model to score well by recognising *who* the subject is rather than *what state they
are in* — an error that inflates the reported performance while telling us nothing about the
intended use case.

Of the 15 subjects, **3 are held out as a test set** and are not used at any point during model
development. The remaining **12 subjects form the development set**, on which
**leave-one-subject-out cross-validation** is performed: each of the 12 folds trains on 11 subjects
(`[FILL: ≈N]` data points) and validates on the single held-out subject (`[FILL: ≈N]` data points).
The reported validation error is the mean across the 12 folds, and the spread across folds is
reported as well.

Cross-validation was preferred over a single split because the dataset is small: with only 12
development subjects, a single held-out subject would give a validation estimate dominated by that
individual's physiology. Averaging over all 12 folds uses every subject for validation exactly once
and, as a by-product, shows how much performance varies from person to person.

---

## 4. Results

The regularisation strength `C` was selected by cross-validated macro-F1 over the grid
{0.001, 0.01, 0.1, 1, 10, 100}, giving C = 1.

| Metric | Training | Validation (mean ± sd over 12 folds) |
|---|---|---|
| Accuracy | 0.916 ± 0.006 | 0.746 ± 0.132 |
| Macro-F1 | 0.875 ± 0.011 | 0.666 ± 0.154 |

`[FILL: 3–5 sentences. State the gap between training and validation error and what it indicates
about over- or underfitting. Report which classes the confusion matrix shows being confused —
`amusement` and `baseline` are the likely pair, since both involve low autonomic arousal. Note the
spread across subjects if it is large.]`

Since only one method is considered at this stage, logistic regression is the final chosen method
by default. Its test error will be reported in Stage 2, once the alternative methods have been
compared and a final model selected; the three held-out subjects remain untouched until then.

---

## 5. Conclusion

In this first stage, we formulated affective state detection as a multi-class classification 
problem and trained an L2-regularised logistic regression model on extracted wearable sensor 
features. Evaluated through leave-one-subject-out cross-validation, the model achieved a mean 
validation Macro-F1 score of 0.666. While this performance indicates that the problem is partially
solved—specifically, the stress condition is highly separable due to distinct physiological 
markers—the overall generalisation to unseen individuals remains poor. As expected from the feature 
distributions, distinguishing mild positive arousal (amusement) from a neutral state (baseline) 
using a linear model is not yet reliable. Consequently, while logistic regression provides a solid 
baseline, it is insufficient to fully capture the nuances of these affective states.

Three limitations stand out. First, the labels are experimental conditions rather than measured
affective states, so a subject who was not actually stressed during the stress condition is
mislabelled by construction. Second, a linear model cannot represent interactions between
physiological signals, such as a change in skin conductance meaning something different depending
on the level of physical activity. Third, with 15 subjects the estimate of between-person
variability is itself uncertain.

Stage 2 will address the second limitation directly by comparing logistic regression against a
random forest and a support vector classifier with an RBF kernel, both of which can represent
non-linear decision boundaries.

---

## Use of AI

In this project, large language models (specifically Claude Opus 5 & Gemini 3.1 pro) were used 
as an assistive tool to accelerate code implementation and refine the academic writing. The core 
conceptualisation, including the selection of the WESAD dataset, the formulation of the multi-class
classification problem, and the decision to employ Logistic Regression as the baseline model, were 
made independently by the author.
During the implementation phase, the AI tool was used to generate the Python scripts for feature 
extraction (windowing and summary statistics), leave-one-subject-out cross-validation, and the 
generation of plots. Furthermore, the AI acted as an analytical assistant: after I observed the 
initial outputs and data structures, the AI helped articulate the interpretations of the visualisations, 
such as identifying collinearity in the correlation heatmap and analysing the misclassifications between 
amusement and baseline in the confusion matrix. Finally, AI was used to polish the structure and tone of 
Sections 3, 4, and 5. All generated code was reviewed, and all results, numeric values, and final 
wording in this report were comprehensively verified and finalised by me.


---

## References

[1] P. Schmidt, A. Reiss, R. Duerichen, C. Marberger and K. Van Laerhoven, "Introducing WESAD, a
Multimodal Dataset for Wearable Stress and Affect Detection," in *Proceedings of the 20th ACM
International Conference on Multimodal Interaction (ICMI)*, 2018, pp. 400–408.


---

## Appendix

Github Link: https://github.com/HawChon/ML_project.git
