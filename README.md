📌 Car Price Prediction using Machine Learning**  

🚀 Project Overview 
This project aims to predict the selling price of a car based on various features such as brand, model, fuel type, transmission type, and more. We use machine learning models to analyze historical data and make accurate price predictions.  

📂 Dataset  
The dataset contains information on used cars with features like:  
- Year (Year of manufacture)  
- Present_Price (Current market value of the car)  
- Selling_Price (Selling price of the used car)  
- Kms_Driven (Total kilometers driven)  
- Fuel_Type (Petrol/Diesel/CNG)  
- Transmission (Manual/Automatic)  
- Owner (Number of previous owners)  

🛠 Tech Stack  
- Programming Language: Python  
- Libraries Used: 
  - `pandas` (Data Handling)  
  - `numpy` (Numerical Computations)  
  - `matplotlib & seaborn` (Visualization)  
  - `scikit-learn` (Machine Learning)  

📊 Data Preprocessing 
- Handling missing values  
- Converting categorical variables using One-Hot Encoding 
- Feature scaling using Standardization/Normalization 
- Splitting the dataset into Training (80%) & Testing (20%) 

📈 Model Training & Evaluation  
We trained multiple models and compared their performance:  

Model                             R² Score
Linear Regression                 0.84      
Random Forest Regressor           0.96 (Best)
Decision Tree Regressor           0.90        |
The Random Forest Regressor performed the best and was selected as the final model.  

🧪 Model Performance Metrics 
To evaluate our model, we used:  
- MAE (Mean Absolute Error) – Measures average error  
- MSE (Mean Squared Error) – Penalizes large errors  
- R² Score – Measures how well the model explains the variance  

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

📌 How to Run the Project? 
1️⃣ Clone the Repository: 
```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

2️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the Jupyter Notebook:  
```bash
jupyter notebook
```

4️⃣ Run Model & Test Predictions:
Modify the `test_data.csv` file and predict prices.

---

📜 Future Improvements 
- Optimize Hyperparameter further  
- Use Deep Learning Models for better accuracy  
- Deploy as a Web App  

📩 Contact & Contributions 
If you find this project useful, feel free to fork & contribute!  
📧 Email: ansarmazhar477@gmail.com  
📌 GitHub: : https://github.com/ansar-mazhar  
