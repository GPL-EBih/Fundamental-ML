# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Lâm Gia Phú|21280104|
    |2|Trần Minh Hiển|21280016|
    |3|Trần Ngọc Khánh Như|21280040|
    |4|Nguyễn Ngọc Thành|21280108|

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push themodels

| Model                | Data Type         | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score |
|----------------------|-------------------|----------|-----------------|--------------|----------------|--------------------|-----------------|--------------------|
| RandomForest         | Original          | 0.470    | 0.580           | 0.410        | 0.430          | 0.500              | 0.470           | 0.450              |
| LogisticRegression   | Original          | 0.362    | 0.410           | 0.300        | 0.290          | 0.350              | 0.360           | 0.340              |
| NaiveBayes           | Original          | 0.212    | 0.220           | 0.220        | 0.170          | 0.270              | 0.210           | 0.200              |
| MLP                  | Original          | 0.421    | 0.430           | 0.400        | 0.410          | 0.430              | 0.420           | 0.420              |
| RandomForest         | PCA-transformed   | 0.406    | 0.630           | 0.330        | 0.360          | 0.540              | 0.410           | 0.370              |
| LogisticRegression   | PCA-transformed   | 0.370    | 0.430           | 0.290        | 0.290          | 0.360              | 0.370           | 0.340              |
| NaiveBayes           | PCA-transformed   | 0.262    | 0.280           | 0.270        | 0.240          | 0.320              | 0.260           | 0.270              |
| MLP                  | PCA-transformed   | 0.432    | 0.440           | 0.420        | 0.420          | 0.430              | 0.430           | 0.430              |


### Original Data
- **Best Model**: Random Forest
- **Performance**: Highest accuracy (0.47), macro and weighted average metrics, indicating consistent performance across classes.

### PCA-Transformed Data
- **Best Model**: MLP
- **Performance**: Highest accuracy (0.432), macro and weighted average metrics, showing balanced and high performance in precision, recall, and F1-score.

### Detailed Results
- **MLP on PCA-Transformed Data**:
  - Most accurate on Emotion 5 (Accuracy: 0.60)
  - Least accurate on Emotion 0 (Accuracy: 0.32)
- **Random Forest on Original Data**:
  - Most accurate on Emotion 5 (Accuracy: 0.67)
  - Least accurate on Emotion 0 (Accuracy: 0.61)

## Conclusion

- **Best Model for Original Data**: Random Forest
- **Best Model for PCA-Transformed Data**: MLP

- The models exhibit their highest performance when categorizing images depicting Emotion 5 (Surprise), but encounter difficulties when classifying images depicting Emotion 0 (Angry).

## Requirements

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib








