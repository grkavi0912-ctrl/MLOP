#for data manipulation
import pandas as pd
import sklearn
import os # Added: import os for os.getenv

#for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#for converting text data into numerical representation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#for hugging face space authentication to upload files
from huggingface_hub import login,HfApi

#Define constants for the dataset and output paths
#api=HfApi(token=os.getenv("HFTOKEN")) # Fixed missing parenthesis
api=HfApi(token=os.getenv("HFTOKEN")) # Corrected missing parenthesis and used os.getenv as in original notebook
DATASET_PATH = "hf://datasets/grkavi0912/Tpro/tour.csv"
df=pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

#Drop the unique identifier
if "UDI" in df.columns:
    df.drop("UDI",inplace=True)
    print("Unique identifier 'UDI' dropped successfully")
else:
    print("Column 'UDI' not found, skipping drop operation.")
# If 'UDI' was a placeholder, you might want to replace it with the actual unique identifier column name if it's different.

#Define a target variable for this classification task
target="ProdTaken"


# Split into features and target
x = df.drop(columns=[target])
y = df[target]


#split into x(features) and y(target)
x=df.drop(columns=[target])
y=df[target]

#perform train and test split
xtrain,xtest,ytrain,ytest = train_test_split(
    x,y, test_size=0.2,random_state=42 # Fixed random_state=42
)

xtrain.to_csv("xtrain.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
xtest.to_csv("xtest.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ("xtrain.csv","xtest.csv","ytrain.csv","ytest.csv")

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="grkavi0912/Tpro",
        repo_type="dataset",
    )
