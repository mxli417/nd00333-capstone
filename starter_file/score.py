import os
import joblib
from pathlib import Path
import pandas as pd 

def init():
  global model
  output_path = Path(os.getenv("AZUREML_MODEL_DIR")) / "outputs"
  assert output_path.exists(), f"Path not found: {output_path.absolute()}"

  model_path = output_path / "model.joblib"
  model = joblib.load(model_path)

def run(raw_data):
  # load data
  data = pd.DataFrame(json.loads(raw_data)["data"])

  # make predictions on the data
  predictions = model.predict(data)
  
  return predictions.tolist()
