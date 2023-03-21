from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler_data")

# Define the threshold for anomaly detection
threshold = 0.05


def detect_anomaly(data):
    # Scale the data
    scaled_data = scaler.transform(data)

    # Reshape the data for the LSTM model
    reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])

    # Make predictions with the model
    predictions = model.predict(reshaped_data)

    # Reshape the predictions and the original data
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
    original_data = data.values.reshape(data.shape[0], data.shape[1])

    # Calculate the mean absolute error between the predictions and the original data
    mae = np.mean(np.abs(predictions - original_data), axis=1)

    # Create a DataFrame with the results
    results = pd.DataFrame({'time': data.index, 'Loss_mae': mae})
    results.set_index('time', inplace=True)

    # Set the anomaly label based on the threshold
    results['Anomaly'] = results['Loss_mae'] > threshold

    return results



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file selected')
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        # Check if the file is valid
        try:
            data = pd.read_csv(file, index_col='time')
        except Exception as e:
            return render_template('index.html', message='Error reading file: {}'.format(str(e)))

        # Detect anomalies in the data
        results = detect_anomaly(data)

        # Generate a plot of the data and anomalies
        plot = results.plot(y='Loss_mae', logy=True, figsize=(16, 9), color=['blue', 'red'], legend=False).get_figure()

        # Save the plot to a file
        #plot.savefig('static/plot.png')
        #plot.clf()

        # Render the results
        return render_template('results.html', data=results.to_html(), plot='plot.png')

    # Render the home page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
