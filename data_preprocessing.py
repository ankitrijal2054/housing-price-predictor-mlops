import pandas as pd
import numpy as np
import os
from scipy.stats import skew

# Load dataset
DATA_PATH = "data/raw/housing.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Dataset Loaded Successfully!")
    print("Shape:", df.shape)
    return df


def handle_missing_values(df):
    
    # Drop column if missing values exceed threshold (40%)
    threshold = 40
    missing_values_percentage = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_values_percentage[missing_values_percentage > threshold].index
    df = df.drop(columns=cols_to_drop)
    
    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values with 'None'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('None')

    print("Missing values handled.")
    return df

def handle_skewness(df):
    # Select numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Compute skewness for each column and filter highly skewed features
    skewness = df[num_cols].apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[skewness > 1].index
    
    # Apply log transformation to highly skewed features except SalesPrice
    for feature in skewed_features:
        if feature == 'SalePrice':
            continue
        df[feature] = np.log1p(df[feature]) 
    
    return df

def handle_collinearity(df):
    # Drop highly correlated features
    corr_matrix = df.drop(columns=["SalePrice"]).corr().abs()  # Exclude SalePrice
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    df = df.drop(columns=to_drop)
    
    return df

def feature_engineering(df):
    # Total SF (Total Square Footage)
    df['Total SF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']

    # House Age, Remodel Age & Garage Age
    df['House Age'] = df['Yr Sold'] - df['Year Built']
    df['Years Since Remodel'] = df['Yr Sold'] - df['Year Remod/Add']
    df['Garage Age'] = df['Yr Sold'] - df['Garage Yr Blt']

    # Total Bathrooms
    df['Total Bathrooms'] = df['Full Bath'] + (df['Half Bath'] * 0.5) + df['Bsmt Full Bath'] + (df['Bsmt Half Bath'] * 0.5)

    # Total Porch Area
    df['Total Porch SF'] = df['Wood Deck SF'] + df['Open Porch SF'] + df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch']
    
    #Encoding Ordinal Variables
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    ordinal_cols = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC']
    df[ordinal_cols] = df[ordinal_cols].apply(lambda col: col.map(qual_map).fillna(0))
    
    # Encoding remaining categorical values with One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Drop original columns used for feature engineering
    drop_feature_cols = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch']
    df.drop(columns=drop_feature_cols, inplace=True)
    
    # Drop Order, PID, Mo Sold and Yr Sold as they are not useful for modeling
    df = df.drop(columns=["Order", "PID", "Mo Sold", "Yr Sold"])
    
    # Handle skewness
    df = handle_skewness(df)
    
    # Handle collinearity
    df = handle_collinearity(df)
    
    print("Feature engineering complete!")
    return df

if __name__ == "__main__":
    df = load_data()
    df = handle_missing_values(df)
    df = feature_engineering(df)

    # Save the processed data
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)
    processed_file = os.path.join(processed_path, "housing_processed.csv")
    
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved at {processed_file}")
