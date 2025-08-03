# TeLIP: Terrain-Based Landslide Initiation Position Model

## 📌 Overview
`TeLIP` is a deep learning model designed to predict landslide initiation positions based on topographic and geologic features of hillslopes. It uses Transformer-based architecture to capture the interdependencies among terrain features along slope profiles. The model is capable of handling complex relationships and providing interpretable outputs for landslide susceptibility assessments.

<!-- The fast assessment of the global minimum adsorption energy (GMAE) between catalyst surfaces and adsorbates is crucial for large-scale catalyst screening. However, multiple adsorption sites and numerous possible adsorption configurations for each surface/adsorbate combination make it prohibitively expensive to calculate the GMAE through density functional theory (DFT). Thus, we designed a novel multi-modal transformer called AdsMT to rapidly predict the GMAE based on surface graphs and adsorbate feature vectors without any binding information. -->
<!-- Three diverse benchmark datasets were constructed for this challenging GMAE prediction task. Our AdsMT framework demonstrates excellent performance by adopting the tailored graph encoder and transfer learning, achieving mean absolute errors of 0.09, 0.14, and 0.39 eV, respectively. Beyond GMAE prediction, AdsMT's cross-attention scores showcase the interpretable potential to identify the most energetically favorable adsorption sites. Additionally, uncertainty quantification was integrated into AdsMT to further enhance its trustworthiness in experimental catalyst discovery. -->


## 🚀 Environment Setup
- System Requirements: This package can run on a standard Windows machine with at least 2 GB of RAM. It does not require a GPU as the code will run on the CPU.
- Install `Anaconda`: Follow the `Anaconda` [installation guide](https://docs.anaconda.com/anaconda/install/).
- Create and activate the environment:
   ```bash
   conda env create -f env.yml
   conda activate telip
   ```
- This will set up the environment with all required dependencies for `TeLIP`.

## 📊 Datasets
Example dataset link: [Zenodo](https://xxx)
- Three types of input data are required for the model: `geological data` as predictors, `landslide position data` as targets, and `slope length data` for evaluating prediction error.
- Users can create their corresponding training datasets based on the following data format according to their landslide research objectives.

1.`Geological data`: The table below provides a reference for `geological data` format Each row represents a position along a slope profile (`Position ID`) within a given sample (`Sample ID`). Each position is characterized by a set of features (`Feature #1` to `Feature #F`), including terrain attributes such as elevation, slope, curvature, and other environmental variables. The number of positions (n) per sample can be adjusted depending on the length and resolution of the slope profile.
| Sample ID | Position ID | Feature #1 |  Feature #2 | ... | Feature #F |
|:---------:|:-----------:|:----------:|:-----------:|:---:|:----------:|
|     0     |      0      |    120.8   |     112     | ... |     1      |
|     0     |      1      |    119.6   |     114     | ... |    0.97    |
|     0     |      2      |    116.3   |     113     | ... |    0.74    |
|    ...    |     ...     |    ...     |     ...     | ... |    ...     |
|     0     |      n      |    32.6    |     108     | ... |    0.65    |
|     1     |      0      |    123.8   |     152     | ... |     1      |
|     1     |      1      |    117.6   |     151     | ... |    0.94    |
|     1     |      2      |    116.3   |     148     | ... |    0.81    |
|    ...    |     ...     |    ...     |     ...     | ... |    ...     |
|     1     |      n      |    26.6    |     128     | ... |    0.62    |
|    ...    |     ...     |    ...     |     ...     | ... |    ...     |
|     N     |      n      |    21.6    |     29      | ... |    0.48    |


2.`Landslide position data`: The `landslide position data` table is structured to track the location of landslide initiation along a slope profile. Each sample (e.g., a slope profile) is divided into multiple positions, with a binary label for each position indicating whether a landslide occurred at that specific location. `Landslide Binary Value` is a binary value indicating landslide occurrence. A 1 indicates that a landslide occurred at that position, and a 0 indicates no landslide at that position.
| Sample ID | Position ID | Landslide Binary Value |
|:---------:|:-----------:|:----------------------:|
|     0     |      0      |            0           |
|     0     |      1      |            0           |
|     0     |      2      |            0           |
|    ...    |     ...     |           ...          |
|     0     |      n      |            0           |
|     1     |      0      |            0           |
|     1     |      1      |            0           |
|     1     |      2      |            1           |
|    ...    |     ...     |           ...          |
|     1     |      n      |            0           |
|    ...    |     ...     |           ...          |
|     N     |      n      |            0           |


3.`Slope length data`: The `slope length data` table provides the total horizontal length of each slope and the true landslide initiation position. Each row represents a unique `Sample ID`, with the `Slope Length` as the total horizontal distance, and the `True Landslide Position` as the horizontal distance from the start of the slope to the landslide location. This data is used to evaluate prediction errors by comparing predicted landslide positions with true positions.
| Sample ID | Slope Length | True Landslide Position |
|:--------:|:------------:|:-----------------------:|
| 0        | 185.3        | 35.0                    |
| 1        | 366.2        | 62.0                    |
| 2        | 284.8        | 26.3                    |
| ...      | ...          | ...                     |
| N        | 495.2        | 54.5                    |

## 🔥 Model Training
### 1. Training from scratch
To train the `SlopeTransformer` model, run the following command:
   ```bash
   python main.py
   ```
This script loads `Geological data.CSV`, `Landslide position data.CSV`, and `Slope length data.CSV`. It trains the model using 5-fold cross-validation, with the `SlopeTransformer` model and `Norm Position Error` (NPE) as the evaluation metric. Training may take several hours, and the final `Mean Norm Position Error` (MNPE) is reported after all folds.

### 2. Cross-Validation for Robust Evaluation
The dataset is split into 5 folds. The model is trained on the training subset and evaluated on the validation subset. The `NPE` metric is used for evaluation, and early stopping is applied to prevent overfitting.

### 3. Model Architecture
The `SlopeTransformer` model uses:
`Positional Encoding` for input feature ordering.
`Transformer Encoder` for feature extraction.
A `Linear Layer` for predicting the landslide position.
The model is trained with `BCELoss` to minimize the error between predicted and actual positions.

## 🌈 Acknowledgements
This work is supported by the National Key R&D Program of China (2024YFC3012601) and the National Natural Science Foundation of China (42277132).

## 📫 Contact
If you have any question or advice, welcome to contact me at:

Junyu Wu: wujunyu@zju.edu.cn



