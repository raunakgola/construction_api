Construction Project Risk Prediction API
========================================

This API provides predictive insights and recommendations for construction projects, including predictions for:

-   **Project Delay**: Estimate the delay in construction based on project parameters.
-   **Cost Overrun**: Predict the probability of cost overrun.
-   **Injury Risk**: Assess injury risks on construction sites.

After making predictions, the API can generate detailed solutions using a language model to mitigate the identified risks.

Table of Contents
-----------------

1.  [Features](#features)
2.  [Setup and Installation](#setup-and-installation)
3.  [Environment Variables](#environment-variables)
4.  [Endpoints](#endpoints)
5.  [Usage](#usage)
6.  [Logging and Rate Limiting](#logging-and-rate-limiting)
7.  [Error Handling](#error-handling)

* * * * *

Features
--------

-   **Predict Delay in Construction Projects**: Receive an estimated delay based on factors like area, height, weather conditions, etc.
-   **Predict Cost Overrun**: Get predictions on possible cost overruns with suggested solutions.
-   **Predict Injury Risk**: Assess the likelihood of injuries based on factors such as working conditions, machine conditions, and primary causes.

Each prediction endpoint also provides recommendations generated by a language model for mitigating the identified risks.

Setup and Installation
----------------------

1.  **Clone the repository**:

    cmd:

    ```
    git clone https://github.com/yourusername/construction_api.git
    ```
    ```cd construction_api```

3.  **Install required packages**: Ensure you have Python installed, then install dependencies using:

    bash

    Copy code

    `pip install -r requirements.txt`

4.  **Load the environment variables**: Create a `.env` file in the root directory with the following environment variables.

Environment Variables
---------------------

Create a `.env` file in the root directory and add the following variables:

plaintext

Copy code

`HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct
HUGGINGFACE_API_KEY=your_huggingface_api_key`

-   `HUGGINGFACE_API_URL`: URL to the Hugging Face API endpoint.
-   `HUGGINGFACE_API_KEY`: Your Hugging Face API key for accessing the LLM model.

Endpoints
---------

1.  **Swagger API Documentation**: [GET] `/custom_swagger`

    -   Provides API documentation in JSON format for integration with Swagger UI.
2.  **Delay Prediction**: [POST] `/delay_predict`

    -   **Description**: Predicts project delay and provides mitigation strategies.
    -   **Body Parameters**:
        -   `features`: Array of numeric values representing project features.
    -   **Response**:
        -   `prediction`: Estimated delay.
        -   `solution`: Generated solution to reduce delay.
3.  **Cost Overrun Prediction**: [POST] `/cost_predict`

    -   **Description**: Predicts cost overrun probability and suggests solutions.
    -   **Body Parameters**:
        -   `features`: Array of numeric values representing project features.
    -   **Response**:
        -   `prediction`: Estimated cost overrun.
        -   `solution`: Suggested solutions to minimize cost.
4.  **Injury Risk Prediction**: [POST] `/injury_predict`

    -   **Description**: Predicts injury risk level and suggests preventive strategies.
    -   **Body Parameters**:
        -   `features`: Array of numeric values representing risk factors.
    -   **Response**:
        -   `prediction`: Predicted risk level.
        -   `solution`: Preventive strategy to reduce injuries.

Usage
-----

### 1\. Delay Prediction

**Request**:

bash

Copy code

`curl -X POST http://localhost:5000/delay_predict -H "Content-Type: application/json" -d '{"features": [feature1, feature2, ...]}'`

**Response**:

json

Copy code

`{
  "prediction": "~10% of the initial completion year",
  "solution": "Solution content generated by the LLM..."
}`

### 2\. Cost Overrun Prediction

**Request**:

bash

Copy code

`curl -X POST http://localhost:5000/cost_predict -H "Content-Type: application/json" -d '{"features": [feature1, feature2, ...]}'`

**Response**:

json

Copy code

`{
  "prediction": "~20% of the initial budget",
  "solution": "Solution content generated by the LLM..."
}`

### 3\. Injury Risk Prediction

**Request**:

bash

Copy code

`curl -X POST http://localhost:5000/injury_predict -H "Content-Type: application/json" -d '{"features": [feature1, feature2, ...]}'`

**Response**:

json

Copy code

`{
  "prediction": "High chances of serious injury",
  "solution": "Solution content generated by the LLM..."
}`

Logging and Rate Limiting
-------------------------

-   **Logging**: Logs are configured using a rotating file handler (`app.log`). Logs important events like requests, errors, and internal server issues.
-   **Rate Limiting**: API requests are rate-limited to 100 requests per minute per IP.

Error Handling
--------------

In case of errors, the API returns a structured JSON response with the error message:

**Response**:

json

Copy code

`{
  "error": "Error message"
}`

Ensure the models (`Delay_model.pkl`, `Cost_Overrun_model.pkl`, `Injury_model.pkl`) are present in the root directory as required for predictions.

* * * * *

License
-------

This project is open source and available under the MIT License.
