# Stock Price Predictor

This repository contains a machine learning model to predict stock prices. The model uses historical stock price data to forecast future prices. 

## Features

- Data preprocessing and feature engineering
- Multiple machine learning algorithms for stock price prediction
- Evaluation metrics to assess model performance
- Visualization of stock price trends and predictions

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Shivcoder27/Stock_price_predictor.git
    cd Stock_price_predictor
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset in CSV format with the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
2. Place your dataset in the `data` directory.
3. Run the preprocessing script to clean and prepare the data:
    ```bash
    python preprocess.py
    ```

4. Train the model using the prepared dataset:
    ```bash
    python train.py
    ```

5. Predict stock prices using the trained model:
    ```bash
    python predict.py
    ```

## Data

The dataset used in this project should contain historical stock prices. You can obtain such data from sources like Yahoo Finance, Google Finance, or other financial data providers.

## Model

The project includes the following machine learning algorithms for stock price prediction:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- LSTM (Long Short-Term Memory) Neural Network

## Evaluation

The performance of the models is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Results

The results of the model training and prediction are visualized using various plots. These include the actual vs. predicted stock prices and error metrics for each model.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to reach out if you have any questions or need further assistance. Happy coding!

