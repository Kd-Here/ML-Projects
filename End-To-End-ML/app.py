from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.logger import logging

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.predict_pipeline import DataCollector,PredictPiple



app = Flask(__name__)


# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        logging.info('Userinput from frontend form is send to backend')
        data = DataCollector(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        logging.info('Userinput is changed into dataframe type ')
        predict_df = data.give_data_as_dataframe()

        predict_piple = PredictPiple()
        result = predict_piple.predict(predict_df)

        return render_template('home.html',res=result[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)        
