import requests
import json
import numpy as np
from predict import app

data = {
    'review': ['العيادة مناسبة ولكن السعر مرتفع']
}

headers = {
    'Content-type': "application/json"
}
# Main code for post HTTP request
url = "http://127.0.0.1:3000/predict"


response = app.test_client().post(
        'predict_api',
         data=json.dumps(data),
        content_type='application/json',
    )

#response = requests.request("POST", url, headers=headers, data=json.dumps(data))

# Show confusion matrix and display accuracy
lambda_predictions = json.loads(response.get_data(as_text=True))
print(lambda_predictions)