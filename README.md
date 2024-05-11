# Real-Time Fraud Detection with Flask (Part-II)

This project focuses on deploying machine learning models for real-time fraud detection using Flask on Vercel. The selected models include Logistic Regression, Decision Tree, Random Forest, and XGBoost. The decision to opt for these models over deep learning alternatives was based on considerations of complexity, performance, and integration feasibility within Flask.

## Project Structure

### Frontend
- **HTML**: Provides the structure of the web pages.
- **CSS**: Utilizes Materialize CSS for styling.
- **JavaScript**: Incorporates Materialize JS for interactive elements.

### Backend
- **Programming Language**: Python
- **Framework**: Flask

The backend processes input data from a user-friendly frontend interface, queries the selected ML model (stored as pickle files), and returns real-time predictions.

### Pickle Library
The Python `Pickle` library is used to serialize and deserialize machine learning models, enabling easy integration within Flask for inference.

### Website Access
The fraud detection system is accessible at [Financial Fraud Detector](https://financialfrauddetector.onrender.com) hosted on Render. Users interact with the frontend to select a machine learning model and specify transaction details, receiving instant fraud likelihood predictions.

## Contributors
- **Saket Kulkarni**
  - [GitHub](https://github.com/StrangeCoder1729)
  - [LinkedIn](www.linkedin.com/in/saketkulkarni1729)
  
- **Tirth Mehta**
  - [GitHub](https://github.com/TirthM21)
  - [LinkedIn](https://www.linkedin.com/in/mehta-tirth/)

## Integration and Deployment

### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Deployment Instructions
To run the project locally:
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the application through the provided localhost URL.

### Website Deployment
The fraud detection system is deployed on Render and can be accessed at [Financial Fraud Detector](https://financialfrauddetector.onrender.com).

This project exemplifies a practical implementation of machine learning for fraud detection in a real-time web application, highlighting Flask's integration with machine learning models for efficient deployment and usability.
