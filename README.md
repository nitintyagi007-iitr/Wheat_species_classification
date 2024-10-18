# Wheat_species_classification

# Nondestructive Identification of Wheat Species using  Deep Convolutional Networks with Oversampling Strategies on Near-Infrared Hyperspectral Imagery

**Authors** :Nitin Tyagi, Sarvagya Porwal, Pradeep Singh, Balasubramanian Raman, Neerja Garg

**Contents**

·         Overview of Project

·         Dataset Description

·         Resources

·         Problem-Solving Approach

·         Results

**Overview of the project**

This project aims to improve the classification of Indian wheat species using near-infrared hyperspectral imaging (NIR-HSI) and deep learning. A dataset of 40 varieties from four species—T. aestivum, T. durum, T. dicoccum, and Triticale—was collected, covering the 900-1700 nm range. To address dataset imbalance, oversampling techniques like SMOTE and ADASYN were used. Classification was performed using a 1D-CNN, 1D-ResNet, and traditional machine learning models (Naive Bayes, KNN, Random Forest, and XGBoost). The 1D-CNN model achieved the highest accuracy of 98.43% with ADASYN, highlighting the effectiveness of NIR-HSI with deep learning for accurate, rapid wheat species identification.

**Dataset Description**

The dataset for this research consists of images of 40 popular wheat varieties harvested in 2022 from certified wheat growers' institutes across five Indian states: Rajasthan, Punjab, Karnataka, Madhya Pradesh, and Haryana. For each variety, 150-200 grams of seeds were collected and stored in plastic bags at 4°C to preserve their condition. Before imaging, seeds were acclimatized to a controlled room environment (25 ± 1°C and 51 ± 5% relative humidity) to ensure consistency. Images were captured using a near-infrared hyperspectral imaging system covering the spectral range of 900-1700 nm, with each wheat seed imaged from both the crease up and crease down orientations. The moisture content of the wheat samples was measured using the ISO 712:2009 oven drying method, yielding values between 12% and 14% on a wet basis. The wheat seed images were acquired using a hyperspectral imaging system operating within the spectral range of 900-1700 nm. Each seed was imaged from both sides—crease up and crease down—to capture comprehensive visual data for analysis.

**Resources**
1. CUDA

2. Python 3.11, PyTorch

3 Scikit-learn, NumPy, Pandas, Rich, Seaborn,

and Matplotlib

**Problem Solving Approach**

![Wheat Seed Image](https://github.com/nitintyagi007-iitr/Wheat_species_classification/blob/main/Traditional%20machine%20learning%20models/Proposed%20approach.png)

**Results**

The classification of the four wheat species is performed using 1D-CNN, 1D-ResNet and four traditional machine learning models namely: Naïve Bayes, KNN, Random Forest and XGBoost. 1D-CNN outperformed the other models. The performance of the models were evaluated using imbalanced and balanced data.
