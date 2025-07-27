# House Price Prediction: A Probabilistic Machine Learning Approach

## Abstract

Predicting house prices is vital in real estate for buyers, sellers, and financial institutions. Traditional methods like Linear Regression often fail to capture complex, nonlinear relationships in housing data. This project leverages a combination of Linear Regression, Decision Trees, and Random Forest models to improve prediction accuracy and minimize errors. The study compares the effectiveness of these models using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Keywords

House Price Prediction, Machine Learning, Linear Regression, Decision Tree, Random Forest, Real Estate

---

## Project Overview

This project aims to predict house prices using machine learning techniques and probabilistic approaches, with a focus on non-linear models like Decision Trees and Random Forests. The dataset includes features such as house price, square footage, bedrooms, location, and amenities, with over 50,000 records sourced from Kaggle/Zillow. Comprehensive preprocessing—including missing value imputation, outlier handling, normalization, and encoding—is performed to prepare the data.

---

## Dataset

- **Source:** Kaggle / Zillow datasets
- **Size:** 50,000+ records
- **Features:** Price, square footage, number of bedrooms, location, facilities
- **Files:**  
  - `train.csv`: Training data (features + target)  
  - `test.csv`: Test data (features or features + target)

---

## Methodology

- **Linear Regression:** A baseline statistical model assuming linear dependencies between features and prices.
- **Decision Tree:** A supervised learning algorithm dividing data into branches based on feature splits to capture non-linear relationships.
- **Random Forest:** An ensemble technique averaging multiple Decision Trees for enhanced accuracy and reduced overfitting.

Models are evaluated using:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)

---

## Data Processing

- Handling missing values using mean/median/mode imputation and advanced methods as applicable.
- Removing duplicate records.
- Normalizing numerical features using Min-Max scaling, Z-Score Standardization, or Robust Scaling.
- Encoding categorical variables via One-Hot Encoding, Label Encoding, or Ordinal Encoding.
- Data split into training and test sets via random or stratified split.

---

## Visualizations

- Correlation heatmaps for Feature analysis.
- Histograms for distribution of key parameters.
- Comparative bar plots illustrating model performance.

*(Figures are included in the notebook and can be viewed there.)*

---

## Results

- Random Forest outperforms Linear Regression and Decision Tree models, showing better accuracy and robustness.
- Demonstrates the value of ensemble methods in capturing complex patterns in real estate data.
- Suggests potential enhancements by incorporating economic indicators and sentiment analysis from external sources.

---

## How to Run the Project

1. Clone this repository:
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction

2. Install required packages:
pip install -r requirements.txt

3. Ensure `train.csv` and `test.csv` are in the project folder.
4. Launch Jupyter Notebook:
jupyter notebook main_project.ipynb

## Requirements

See `requirements.txt` file for all necessary Python packages, typically:


pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter

---

## Future Work

- Inclusion of macroeconomic variables such as interest rates and inflation.
- Sentiment analysis using real estate news and social media data.
- Exploration of advanced machine learning/deep learning models.

---

## References

1. S. B. Jha et al., "Machine learning approaches to real estate market prediction problem: A case study," arXiv:2008.09922, 2020.  
2. K. Xu and H. Nguyen, "Predicting housing prices and analyzing real estate market in the Chicago suburbs using machine learning," arXiv:2210.06261, 2022.  
3. H. Sharma, H. Harsora, and B. Ogunleye, "An optimal house price prediction algorithm: XGBoost," arXiv:2402.04082, 2024.  
*(Additional references as needed)*

---

