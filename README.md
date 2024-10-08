# Census Income Predictor - ML Model Deployment

## Overview
This project deploys a machine learning model that predicts income levels based on census data. It uses FastAPI for creating a RESTful API and Docker for containerization, ensuring easy deployment and scalability.

## Features
- Random Forest classifier for income prediction
- FastAPI for efficient API creation
- Docker containerization for consistent deployment
- Swagger UI for interactive API documentation

## Prerequisites
- Python 3.9+
- Docker
- Git

## Installation & Setup

### Local Setup
1. Clone the repository:
git clone https://github.com/your-username/census-income-predictor.git
cd census-income-predictor


2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies:
pip install -r requirements.txt


4. Train the model (if not pre-trained):
python model.py


5. Run the FastAPI server:
uvicorn main:app --reload


6. Access the API at `http://127.0.0.1:8000`

### Docker Setup
1. Build the Docker image:
docker build -t census-income-predictor .


2. Run the Docker container:
docker run -d -p 8000:80 census-income-predictor


3. Access the API at `http://localhost:8000`

## Usage
- API Documentation: Visit `/docs` or `/redoc` for interactive API documentation.
- Make predictions:
  - Send a POST request to `/predict` with the required input features.
  - Example using curl:
    ```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [39, 77516, 13, 2174, 0, 40, ...]}'

## Project Structure
census-income-predictor/
│
├── data/
│   ├── initial/
│   │   └── census-income.csv
│   ├── encoded/
│   └── processed/
│
├── static/
│   └── favicon.ico
│
├── main.py
├── model.py
├── Dockerfile
├── requirements.txt
└── README.md


## API Endpoints
- GET `/`: Welcome message
- GET `/docs`: Swagger UI for API documentation
- GET `/redoc`: ReDoc UI for API documentation
- POST `/predict`: Make income predictions

## Model Information
- Algorithm: Random Forest Classifier
- Features: [List key features used in the model]
- Target: Income level (<=50K, >50K)
- Performance Metrics: [Include accuracy, precision, recall, F1-score]

## Development
- The `model.py` file contains the code for data preprocessing, model training, and evaluation.
- `main.py` sets up the FastAPI application and defines the API endpoints.
- The Dockerfile specifies the container configuration for deployment.

## Contributing
Contributions to improve the project are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request
