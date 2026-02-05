import pandas as pd
import sklearn
#for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
#for model training,tuning and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report,recall_score, precision_score, f1_score, confusion_matrix
#for model serialization
import joblib
#for Hugging Face API
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError # Corrected import
import os # Ensure os is imported for os.getenv if needed
import mlflow # Ensure mlflow is imported

# Define DATASET_PATH here, similar to previous cells
DATASET_PATH = "hf://datasets/grkavi0912/Tpro/tour.csv"

tour_dataset=pd.read_csv(DATASET_PATH) # Changed this line to use DATASET_PATH
print("Dataset loaded successfully")

df=tour_dataset.copy()

#Drop the unique identifier
if "UDI" in df.columns:
    df.drop("UDI",inplace=True)
    print("Unique identifier 'UDI' dropped successfully")
else:
    print("Column 'UDI' not found, skipping drop operation.")

# Drop 'Unnamed: 0' and 'CustomerID' as they are typically identifiers and not features
columns_to_drop = []
if "Unnamed: 0" in df.columns:
    columns_to_drop.append("Unnamed: 0")
if "CustomerID" in df.columns:
    columns_to_drop.append("CustomerID")

if columns_to_drop:
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped identifier columns: {columns_to_drop}")

#Define a target variable for this classification task
target="ProdTaken"

#split into x(features) and y(target)
x=df.drop(columns=[target])
y=df[target]

#perform train and test split
xtrain,xtest,ytrain,ytest = train_test_split(
    x,y, test_size=0.2,random_state=42
)

#set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define numeric and categorical features
numeric_features = xtrain.select_dtypes(include=['number']).columns.tolist()
categorical_features = xtrain.select_dtypes(include=['object', 'category']).columns.tolist()

#Define the preprocessing steps
preprocessor = make_column_transformer(
     (StandardScaler(),numeric_features),
     (OneHotEncoder(handle_unknown='ignore'),categorical_features)
)

#Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight,random_state=42)

#Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [2, 4],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__colsample_bylevel': [0.8, 1.0],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.6]
}

#model pipeline
model_pipeline = make_pipeline(preprocessor,xgb_model)

with mlflow.start_run():
    #Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline,param_grid,cv=5,n_jobs=-1)
    grid_search.fit(xtrain,ytrain)

    #log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        params_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        #log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
             mlflow.log_params(params_set)
             mlflow.log_metric('mean_test_score',mean_score)
             mlflow.log_metric('std_test_score',std_score)

   #log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

  #store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold =0.45

    y_pred_train_proba = best_model.predict_proba(xtrain)[:,1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(xtest)[:,1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metric("train_accuracy", train_report['accuracy'])
    mlflow.log_metric("train_recall", train_report['1']['recall'])
    mlflow.log_metric("train_precision", train_report['1']['precision'])
    mlflow.log_metric("train_f1_score", train_report['1']['f1-score'])
    mlflow.log_metric("test_accuracy", test_report['accuracy'])
    mlflow.log_metric("test_recall", test_report['1']['recall'])
    mlflow.log_metric("test_precision", test_report['1']['precision'])
    mlflow.log_metric("test_f1_score", test_report['1']['f1-score'])

    #save the model locally
    model_path = "best_tour_model.joblib"
    joblib.dump(best_model, model_path)

    # log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at {model_path}")

    #upload to hugging face
    repo_id = "grkavi0912/Tpro"
    repo_type="model"

    # Initialize API client within this cell
    api = HfApi(token=os.getenv("HFTOKEN"))

    #step 1:check if the space exists
    try:
        api.repo_info(repo_id=repo_id,repo_type=repo_type)
        print(f"Repo {repo_id} already exists")
    except RepositoryNotFoundError:
        print(f"Repo {repo_id} does not exist, creating...")
        api.create_repo(repo_id=repo_id,repo_type=repo_type)
        print(f"Repo {repo_id} created successfully")

    #create_repo("tour_model",repo_type="model",private=False)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Model uploaded to {repo_id}")
