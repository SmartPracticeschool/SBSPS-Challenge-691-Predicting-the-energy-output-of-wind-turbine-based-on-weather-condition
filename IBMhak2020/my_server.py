from flask import Flask, redirect, url_for, jsonify, request, render_template
import pandas as pd
# from tensorflow.keras.models import model_from_json

app = Flask(__name__)  
 
# @app.route('/current_weather', methods = ['POST'])
# def current_weather():
# 	# lat long of turkey, talova
# 	lat_long = "40.6549,29.2842"
# 	api_key = "427d13e33d9f4be58db133953201506"

# 	# api for getting current weather data
# 	url = "http://api.worldweatheronline.com/premium/v1/weather.ashx?key=" + api_key + "&q=" + lat_long + "&num_of_days=1&format=json"
# 	if request.method == 'POST':
# 		weather_data = requests.get(url)
	
# 	return jsonify(weather_data.json())

@app.route('/',methods = ['GET']) 
def show_dataset(): 
	df = pd.read_csv('integrated_data.csv')
	df_values = df.values
	return render_template('my_form.html', data = df_values)
  
if __name__ == '__main__': 
   app.run(debug = True) 