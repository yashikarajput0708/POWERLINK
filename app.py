from flask import Flask, render_template, redirect, request,flash
import mysql.connector
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

app = Flask(__name__)

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='powerplant'
)

mycur = mydb.cursor()




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber= request.form['phonenumber']
        age  = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password,`phone number`,age) VALUES (%s, %s, %s, %s,%s)'
                val = (name, email, password, phonenumber,age)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
               msg = 'user logged successfully'
               return redirect("/upload")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')
                            
# Set a secret key for session management (for flash messages)
app.secret_key = 'bhuvana'

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if the file extension is allowed (CSV only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        # If file is selected and is CSV
        if file and allowed_file(file.filename):
            # Save the file to the upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Read the CSV file
            dataset = pd.read_csv(filename)
            dataset_dict = dataset.to_dict(orient='records')

            # Flash a success message
            flash('CSV file uploaded successfully!', 'CSV file uploaded successfully!')

            return render_template('upload.html', dataset=dataset_dict)

        else:
            flash('Only CSV files are allowed!', 'error')
            return redirect(request.url)

    return render_template('upload.html', dataset=None)







# Load models and scalers
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('linear_regression_model.pkl')
xgb_model = joblib.load('xgb_best_model.pkl')
lstm_model = load_model('lstm_model.h5')
scaler_y = joblib.load('scaler_y.pkl')

# Feature engineering function
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.month
    df['year'] = df.index.year
    df['season'] = df['month'] % 12 // 3 + 1
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    # Additional features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
    df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = (df['dayofmonth'] == 1) & (df['month'] % 3 == 1).astype(int)
    df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max'))
    
    # Additional features
    df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)
    
    # Minute-level features
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']
    
    return df.astype(float)

# Flask route for the model prediction
@app.route('/algo', methods=['GET', 'POST'])
def algo():
    mse = None
    mae = None

    # Load the dataset
    df = pd.read_csv("uploads/powerconsumption.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Create features for the dataframe
    df = create_features(df)

    # Prepare features and target
    X = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
    y = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
    
    # Preprocess target values
    y_scaled = scaler_y.transform(y)

    if request.method == 'POST':
        selected_model = request.form['model_selection']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.25, shuffle=False)

        # Model selection and prediction
        if selected_model == 'rf':
            y_pred = rf_model.predict(X_test)
        elif selected_model == 'lr':
            y_pred = lr_model.predict(X_test)
        elif selected_model == 'xgb':
            y_pred = xgb_model.predict(X_test)
        elif selected_model == 'lstm':
            # Reshape the data for LSTM (make it 3D)
            X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Use the pre-trained LSTM model to make predictions
            y_pred = lstm_model.predict(X_test_lstm)
        
        # # Inverse transform predictions to get the original scale
        # y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        # y_test_rescaled = scaler_y.inverse_transform(y_test)

        # Calculate MSE and MAE
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    return render_template('algo.html', mse=mse, mae=mae)



# Load the trained XGBoost model and scaler
xgb_model = joblib.load("xgb_best_model.pkl")  
scaler_y = joblib.load("scaler_y.pkl")  

# Function to create features from input data (same as in your original code)
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.month
    df['year'] = df.index.year
    df['season'] = df['month'] % 12 // 3 + 1
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
    df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = (df['dayofmonth'] == 1) & (df['month'] % 3 == 1).astype(int)
    df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max'))
    df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']
    return df.astype(float)

# Function to preprocess the input data
def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    input_df['Datetime'] = pd.to_datetime(input_df['Datetime'])
    input_df = input_df.set_index('Datetime')
    input_df = create_features(input_df)
    return input_df

# Function to predict power consumption
def predict_power_consumption(input_data):
    input_df = preprocess_input(input_data)
    X_input = input_df.values
    y_pred = xgb_model.predict(X_input)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(1, -1))
    return y_pred[0]

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {
            'Datetime': request.form['datetime'],
            'Temperature': float(request.form['temperature']),
            'Humidity': float(request.form['humidity']),
            'WindSpeed': float(request.form['windspeed']),
            'GeneralDiffuseFlows': float(request.form['general_diffuse_flows']),
            'DiffuseFlows': float(request.form['diffuse_flows'])
        }
        
        # Get the predicted power consumption
        predicted_consumption = predict_power_consumption(input_data)
        
        # Display the prediction result in the template
        msg = f"Predicted Power Consumption: Zone 1: {predicted_consumption[0]:.2f}, Predicted Power Consumption-Zone 2: {predicted_consumption[1]:.2f}, Predicted Power Consumption-Zone 3: {predicted_consumption[2]:.2f}"
        return render_template('prediction.html', msg=msg)
    
    # If GET request, just display the empty form
    return render_template('prediction.html', msg=None)


if __name__ == '__main__':
    app.run(debug=True)
