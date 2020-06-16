from flask import Flask, redirect, url_for, jsonify, request, render_template

app = Flask(__name__)  

def load_model():
	from tensorflow.keras.models import model_from_json
	print("Loading model")
	with open('wind_turbine_architecture.json', 'r') as f:
		model = model_from_json(f.read())

	model.load_weight('model_with_2lstm100_1dense128_dropout0.25_layers.h5')
	return model

def fetch_and_normalize_data_and_predict_data(number):
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.metrics import mean_squared_error

	print("In fetch_and_normalize_data_and_predict_data")
	n_steps_in = 216
	n_steps_out = 72

	df = pd.read_csv('integrated_data.csv')
	df.drop('end_date', axis = 1, inplace = True)

	print("Loaded the file")
	sc = MinMaxScaler()
	scaled_data = sc.fit_transform(df.values[(number - n_steps_in) : number, :])
	scaled_x = scaled_data[(number - n_steps_in) : number, :-1]
	scaled_y = scaled_data[number: (number + n_steps_out), -1] 
	
	print("scaling done!!!")
	model = load_model()
	print("Model loaded!!")
	yhat = model.predict(scaled_x.reshape(1, 216, 10))
	y_hat_reshaped = yhat.reshape(yhat.shape[1])
	return  y_hat_reshaped, scaled_y, mean_squared_error(y_hat_reshaped, scaled_y)


@app.route('/',methods = ['GET']) 
def about(): 
	return render_template('about.html')

@app.route('/evaluate',methods = ['GET']) 
def evaluate(): 
	return render_template('evaluate.html')

@app.route('/show_result', methods = ['POST'])
def show_result():
	if request.method == "POST":
		print("In show show_result function")
		number = int(request.form['number'])
		prediction, true, error = fetch_and_normalize_data_and_predict_data(number)
		return error

if __name__ == '__main__': 
   app.run(debug = True) 