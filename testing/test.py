import numpy as np
from apis import app
from fastapi.testclient import TestClient



client = TestClient(app)



def test_home_endpoint():
    
    response = client.get('/')

    assert response.status_code == 200
    assert "message" in response.json()




def test_health_endpoint():
    
    response = client.get('/health')

    assert response.status_code == 200
    assert "status" in response.json()





def test_predict_endpoint():

    # Generate dummy flat list of 63 landmark values
    dummy_landmarks = list(np.random.rand(63))

    # Send the request with proper structure
    response = client.post('/predict', json={"landmarks": dummy_landmarks})

    # Check response
    assert response.status_code == 200
    assert "label" in response.json()


