from flask import Flask, redirect, url_for, jsonify, request, render_template
import os

app = Flask(__name__)  

EPOCHS = 36
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUMBER_OF_STEPS_IN = 144
NUMBER_OF_STEPS_OUT = 72
CLIP_NORM = 0.5
FEATURES = 6

def load_model():

	'''
		Args: no args
		Return pretrained model

		What it does?
		loads the model and returns it
	'''
	from tensorflow.keras.models import model_from_json
	# print("Loading model")
	with open('wind_turbine_architecture_20_6.json', 'r') as f:
		model = model_from_json(f.read())

	model.load_weight('wind_turbine_weights_20_6.h5')
	return model

def fetch_and_normalize_data(number):
	'''
		Args: number(int): which it gets from evaluate tab in navbar
		Returns the scaled_x (numpy array) and scaled_y(numpy array)

		what it does?
		Takes that number fetches the previous n_steps from dataset and 
		returns the scaled previous n_steps and next n_steps_out for evaluation.
	'''
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.metrics import mean_squared_error

	print("In fetch_and_normalize_data_and_predict_data")
	# n_steps_in = 144
	# n_steps_out = 72
	# n_features = 6

	df = pd.read_csv('model_data.csv')

	print("Loaded the file")
	sc = MinMaxScaler()
	scaled_data = sc.fit_transform(df.values[(number - NUMBER_OF_STEPS_IN) : (number + NUMBER_OF_STEPS_OUT), :])

	scaled_x = scaled_data[:NUMBER_OF_STEPS_IN,  :-1]
	scaled_y = scaled_data[-NUMBER_OF_STEPS_OUT: , -1] 
	
	print("scaling done!!!")

	return scaled_x, scaled_y


def fetch_last_n_stepsin(record):
	'''
		Args: record (a list)
		Returns: last (n_steps _in - 1) of the dataset and concats with the new 
		record enterd by the user in predict tab of navbar.
		return type: numpy array 
	'''
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	import numpy as np

	# n_steps_in = 144
	# n_steps_out = 72
	# n_features = 6

	df = pd.read_csv('model_data.csv')
	print("Loaded the file")

	sc_x = MinMaxScaler()
	data = df.values[-(NUMBER_OF_STEPS_IN - 1):, :-1]
	record = np.array(record)
	X = np.concatenate( (data, record.reshape(1, record.shape[0]) ), axis = 0)

	return sc_x.fit_transform(X) 


def predict_label(x, y = None):
	'''
		Args: X (numpy array), y(numpy array)
		Returns: prediction(numpy array), y(numpy array) if provided as argument.

		What it does?
		Predicts the output for given features
	'''
	# n_steps_in = 144
	# n_steps_out = 72
	# n_features = 6

	model = load_model()
	print("Model loaded!!")
	yhat = model.predict(x.reshape(1, NUMBER_OF_STEPS_IN, n_features))
	y_hat_reshaped = yhat.reshape(yhat.shape[1])
	
	return  y_hat_reshaped, y

# def save_record_to_csv(record):
# 	import pandas as pd

# 	df = pd.read_csv('model_data.csv')
# 	df.loc[-1] = record

# 	print('record saved successfully')

# def create_plot(prediction, true = None):
# 	'''
# 		Args: prediction, true(both numpy arrays)
# 		Returns: graph to be plotted in UI

# 		What it does?
# 		Plots the prediction and ground truth if provided using plotly.
# 	'''
# 	import plotly
# 	import plotly.graph_objects as go
# 	import json

# 	fig = go.Figure()

# 	fig.add_trace(
# 	    go.Scatter(
# 	        x=prediction,
# 	        y=list(range(72)),
# 	        mode='lines', name='prediction',
#                     opacity=0.8, marker_color='orange'
# 	    ))

# 	if true is not None:
# 		fig.add_trace(
# 		    go.Scatter(
# 		        x=true,
# 		        y=list(range(72)),
# 		        mode='lines', name='True data',
# 	                    opacity=0.8, marker_color='blue'
# 		    ))

# 	fig.show()
# 	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# 	return graphJSON

def create_plot(prediction, true = None):
	import matplotlib.pyplot as plt 

	plt.plot(list(range(NUMBER_OF_STEPS_OUT)), prediction, label = "Prediction")

	if true is not None:
		plt.plot(list(range(NUMBER_OF_STEPS_OUT)), true, label = "Ground truth")

	plt.title("Hours v/s Energy")
	plt.legend()

@app.route('/',methods = ['GET']) 
def index():
	'''
		Args: no args
		Returns the home page of application
	'''

	data = {
			'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'lr':LEARNING_RATE,
			'n_steps_in':NUMBER_OF_STEPS_IN,'n_steps_out':NUMBER_OF_STEPS_OUT,
			'clip_norm':CLIP_NORM
			}
	return render_template('about.html', data = data)


@app.route('/evaluate',methods = ['GET', 'POST']) 
def evaluate(): 
	'''
		Args: no args:
		If get request:
		   returns the evaluate template
		else:
			fetches the number and passes it to fetch_and_normalize_data() as 
			parameter which returns X, y and that X,y  are passed to predict_label()
			as arguments and then that predictions are plotted using create_plot() 
			function.
	'''
	if request.method == 'GET':
		return render_template('evaluate.html')
	else:
		number = int(request.form['number'])
		print("number", number)
		scaled_x, scaled_y = fetch_and_normalize_data(number)
		prediction, true = predict_label(scaled_x, scaled_y)

		create_plot(prediction, true)
		data = {
			'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'lr':LEARNING_RATE,
			'n_steps_in':NUMBER_OF_STEPS_IN,'n_steps_out':NUMBER_OF_STEPS_OUT,
			'clip_norm':CLIP_NORM
			}
		return render_template('about.html', data = data)

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	'''
		Args: no args:
		If request if GET:
		   returns the template for individual prediction
		else:
			takes all inputs from user and then passes it as argument to 
			fetch_last_n_stepsin() as parameter and then the values are predicted 
			using predict_label() function and plotted using create_plot().
	'''
	if request.method == 'GET':
		return render_template('prediction.html')
	else:
		wind_speed = float(request.form['wind_speed'])
		tpc = float(request.form['tpc'])
		wind_d = float(request.form['wind_d'])
		wind_gust = float(request.form['wind_gust'])
		dew_point = float(request.form['dew_point'])
		wind_chill = float(request.form['wind_chill'])

		record = [wind_speed, tpc, wind_d, wind_gust, dew_point, wind_chill]
		X = fetch_last_n_stepsin(record)
		prediction, y = predict_label(X)

		create_plot(prediction)

		data = {
			'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'lr':LEARNING_RATE,
			'n_steps_in':NUMBER_OF_STEPS_IN,'n_steps_out':NUMBER_OF_STEPS_OUT,
			'clip_norm':CLIP_NORM
			}

		return render_template('about.html', data = data)

if __name__ == '__main__':
	app.run(debug = False)
