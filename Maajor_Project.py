import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Correct the file path
file_path = r'C:\Users\hp\Downloads\OnlineRetail (1) (1).xlsx'

# Load the dataset
data = pd.read_excel(file_path)

# Data preprocessing
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['TotalAmount'] = data['Quantity'] * data['UnitPrice']

# Scaling the TotalAmount to a typical rating scale (0.5 to 5)
scaler = MinMaxScaler(feature_range=(0.5, 5))
data['TotalAmount'] = scaler.fit_transform(data[['TotalAmount']])

# Verify the data
print(data[['CustomerID', 'StockCode', 'TotalAmount']].head())

# Load data into Surprise library
reader = Reader(rating_scale=(0.5, 5))
data_surprise = Dataset.load_from_df(data[['CustomerID', 'StockCode', 'TotalAmount']], reader)

# Train-test split
trainset, testset = train_test_split(data_surprise, test_size=0.2)

# Train the SVD model
svd = SVD()
cross_validate(svd, data_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Fit on the whole training set
trainset = data_surprise.build_full_trainset()
svd.fit(trainset)

# Make predictions on the test set
predictions = svd.test(testset)

# Calculate RMSE manually
true_ratings = [pred.r_ui for pred in predictions]
predicted_ratings = [pred.est for pred in predictions]
mse = mean_squared_error(true_ratings, predicted_ratings)
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')
