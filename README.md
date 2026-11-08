# Classifying Plastics Using NIR Hyperspectral Imagery

This repo contains a machine learning project I conducted individually to further my understanding of near-infrared (NIR) spectroscopy and the implementation of machine learning (ML) workflows to analyze NIR hyperspectral imagery. For this project, I used a publicly available labeled NIR hyperspectral dataset to build and compare machine learning frameworks for the task of classifying types of plastic based on NIR signatures.

## Dataset

This project uses [**Near-infrared reflection spectrum of plastics**](https://doi.org/10.17632/jg2z4g6pdh.1) by Jeon, Yoo, and Um (2022), collected in a plastic recycling facility.

The workbook contains:

- **224 spectral channels**, covering approximately **933–1720 nm**.
- Five labelled categories: **PE, PP, PET, PS, and PVC**.
- Lateral and driving coordinates describing sampled positions.
- Wavelength information, dark-current reference measurements, and separate training and noisy test sets.

## Questions

1. **Can I visually identify the category of plastic myself by looking at the NIR spectral patterns?**
2. **How does the accuracy of simple and more complex machine learning classifiers compare?** Do neural networks provide an advantage over simpler methods?

## Part 1: Exploring the Dataset

With little understanding of the organization of the dataset, I first mapped the training coordinates to understand the spatial structure behind the spectral measurements. The 50,000 measurements occupied **8,905 unique coordinate positions**, with repeated positions and substantial gaps across the overall coordinate bounds.

Grouping neighbouring positions identified **60 spatial clusters**, each likely corresponding to a unique plastic object. Every cluster contained measurements from just one plastic category: **100% label purity**, consistent with 60 separate objects.

<img width="2370" height="2974" alt="imagePlasticTypes" src="https://github.com/user-attachments/assets/691e5b84-c51e-4473-9856-837910c50d12" />

*Each colour identifies a plastic category. Spatial proportions were stretched for readability.*

Coordinates helped explore the dataset and define validation groups. **They were excluded from all model inputs.**

## Part 2: Visually Comparing Spectral Signatures

After subtracting the dark-current magnitude provided in the dataset from each frequency channel, I calculated and plotted the average spectra for each category, shading in the variance of the intracategorical spectral data.

<img width="2970" height="1759" alt="PE_dark_corrected_normalized_variability" src="https://github.com/user-attachments/assets/31068c6b-05cf-45f1-8c3a-3365e00fcd26" />

<img width="2970" height="1759" alt="PP_dark_corrected_normalized_variability" src="https://github.com/user-attachments/assets/057627ef-7957-44b9-a889-51a79038a803" />

<img width="2970" height="1759" alt="PET_dark_corrected_normalized_variability" src="https://github.com/user-attachments/assets/45fb3495-7823-4f88-9edf-788d38cb7124" />

<img width="2970" height="1759" alt="PS_dark_corrected_normalized_variability" src="https://github.com/user-attachments/assets/1606b907-4848-4780-bc87-cb535d4241ec" />

<img width="2970" height="1759" alt="PVC_dark_corrected_normalized_variability" src="https://github.com/user-attachments/assets/f9e0c762-121a-4902-95cc-74590e58323b" />

While each plastic's NIR signature revealed a similar pattern, there appeared to be enough unique markers that I could use to visually identify a plastic by its spectrum. To check this, I overlaid each category's average spectrum and highlighted unique features I could use to identify them.

<img width="3934" height="2734" alt="average_spectra_overlay_with_features" src="https://github.com/user-attachments/assets/ad102969-7924-4a37-ba3f-5b123dd75e91" />

Upon overlaying the different NIR signatures, I made the following observations tied to the chemical characteristics of each of the plastics:

- PE was the easiest to recognize, with deep dips around **1214 nm** and **1419 nm**, likely resulting from its high C–H makeup.
- PP was similar but had shallower dips than PE, with a lower relative signal around **1110–1195 nm**.
- PET had a steep dip near **1130 nm**, as well as a distinctive feature near **1663 nm**.
- PS had a deep dip near **1144 nm** and an identifiably lower signal at the high-wavelength end.
- PVC was the hardest to distinguish; however, the dips prominent in other categories were less pronounced.

## Part 3: Comparing Classification Models

To better understand the complexities of working with NIR hyperspectral imagery in a machine learning context, I decided to test machine learning frameworks at three different levels of sophistication.

### 1. Spectral Threshold Voting

A weighted vote classifier with learned thresholds for discerning high absorption levels at various wavelength regions.

- I selected candidate regions based on my observations from visually inspecting the NIR signatures earlier.
- Local dip depths were measured relative to neighbouring wavelengths.
- Decision-tree stumps learned feature thresholds.
- Logistic regression learned category-specific vote weights.

<img width="1743" height="1765" alt="confusion_matrix" src="https://github.com/user-attachments/assets/64b28970-0df4-48e7-9cf8-cc932a69eaff" />

**Test accuracy: 91.034%**

### 2. Full-Spectrum Logistic Regression

I chose a multidimensional logistic regression algorithm with the hypothesis that it would do a great job of classifying the plastics. Given that I was visually able to discern the different plastic types by their NIR signatures, I reasoned that a linear classifier fine-tuned to the intricacies of their spectra would be very successful.

<img width="1743" height="1765" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8d397d51-a45c-4b8f-a4e9-6c742abf4277" />

This model achieved the **best test performance** among the configurations explored. The only misclassifications involved false negatives and false positives related to PVC, which I found the hardest to discern visually.

**Test accuracy: 99.995%**

### 3. PyTorch Neural Network

Finally, I set out to test whether a neural network would trump all in terms of classification prowess. Since model architecture leaves a lot of room for design variation, I decided to use a very standard model setup with the assumption that it would provide generalizable results:

- Two different model architectures were tested: **224 → 64 → 32 → 5** (two hidden layers) and **224 → 32 → 5** (one hidden layer).
- ReLU activation, the Adam optimizer, and cross-entropy loss were used.
- Validation selected the training duration; the final model trained for **47 epochs**.

**Two hidden layers:**

<img width="1743" height="1765" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d17d69c6-4bb8-49b3-9a75-797d6d681f8e" />

**Accuracy: 99.921%**

**One hidden layer:**

<img width="1743" height="1765" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b517139b-e23f-4d02-81a9-87b1e9f5b990" />

**Accuracy: 99.878%**

### Test Results

Precision, recall, and F1 are macro averages, giving each plastic category equal weight.

| Model | Accuracy | Precision | Recall | F1 |
| ------------------------- | --------: | --------: | --------: | --------: |
| Spectral threshold voting | 91.03% | 91.11% | 91.49% | 91.04% |
| Logistic regression | 99.95% | 99.95% | 99.95% | 99.95% |
| Neural network: 64 → 32 | 99.92% | 99.92% | 99.93% | 99.92% |
| Neural network: 32 | 99.88% | 99.87% | 99.89% | 99.88% |

<img width="2970" height="1784" alt="model_accuracy_comparison" src="https://github.com/user-attachments/assets/e914e0b0-34ae-467a-b64c-56e63b8a4355" />

## Conclusions

**Can I distinguish plastics visually?**

Yes—upon analysis, I found that the category-average spectra showed recognizable patterns. PE’s deep dips near 1214 and 1419 nm, PET’s feature near 1663 nm, and PS’s dip near 1144 nm were particularly useful. This was an exploratory visual assessment rather than a formal blind identification test.

**How much model complexity is needed?**

All models successfully classified plastics. Full-spectrum logistic regression and both neural networks substantially outperformed the seven-feature voting model. Logistic regression performed best, while the one- and two-layer networks produced similar results.

## Dataset Reference

Jeon, Y., Yoo, J., & Um, S. (2022). *Near-infrared reflection spectrum of plastics* (Version 1) [Data set]. Mendeley Data. [https://doi.org/10.17632/jg2z4g6pdh.1](https://doi.org/10.17632/jg2z4g6pdh.1)
