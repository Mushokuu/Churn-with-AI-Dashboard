# AI-Powered Customer Churn Prediction Dashboard

This project demonstrates a customer churn prediction system using machine learning and an interactive dashboard powered by Streamlit. The system leverages ensemble models such as Random Forest, AdaBoost, and Gradient Boosting to predict the likelihood of customer churn.

## Features
- **Churn Prediction:** Predict customer churn using Random Forest, AdaBoost, and Gradient Boosting.
- **Interactive Dashboard:** Built using Streamlit for visualizing customer churn predictions and insights.
- **Model Explainability:** Integrated SHAP for interpreting the model’s decisions and visualizing feature importance.
- **Real-time Predictions:** The system supports dynamic predictions based on user input for real-time churn forecasting.

## Technologies Used
- **Python**
- **Streamlit** for building the interactive dashboard.
- **Random Forest**, **AdaBoost**, **Gradient Boosting** for classification models.
- **SHAP** for model interpretability.
- **Pandas**, **NumPy** for data manipulation.
- **Scikit-learn** for machine learning.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Mushokuu/Churn-with-AI-Dashboard.git
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## How It Works

1. The user inputs customer details into the dashboard.
2. The system processes the input, and the ensemble models predict the likelihood of customer churn.
3. The predictions are displayed along with key insights about the most influential features using SHAP values.

## Folder Structure
- `app.py`: Main Streamlit application file.
- `model.py`: Contains the machine learning models and prediction functions.
- `data/`: Directory containing datasets (note: large datasets should be handled separately).
- `requirements.txt`: List of dependencies.
- `.gitignore`: Excludes unnecessary files from version control.

## Contributing

Feel free to fork the repository and create pull requests with improvements or bug fixes. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thank you to the authors of the libraries used in this project.
