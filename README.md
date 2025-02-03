# House Price Prediction

This project aims to predict house prices using machine learning algorithms. The dataset used for this project contains various features of houses, such as the number of bedrooms, bathrooms, square footage, and more.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/house_price_prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd house_price_prediction
    ```
3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the dataset:
    - Place your dataset in the `data/` directory.
    - Ensure the dataset is in CSV format and named `indian_house_prices.csv`.

2. Run the preprocessing script:
    ```bash
    python scripts/preprocess_data.py
    ```

3. Train the model:
    ```bash
    python scripts/train_model.py
    ```

4. Make predictions:
    ```bash
    python scripts/predict.py --input data/new_house_data.csv --output predictions.csv
    ```

## Project Structure

```
house_price_prediction/
├── data/
│   ├── house_prices.csv
│   └── new_house_data.csv
├── models/
│   └── house_price_model.pkl
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── predict.py
├── README.md
└── requirements.txt
```

## Model Training

The model training script (`house_price_prediction.py`) reads the preprocessed data, trains a machine learning model, and saves the trained model to the `models/` directory.

## Prediction

The prediction script (`app.py`) loads the trained model and uses it to make predictions on new data. The predictions are saved to the specified output file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.