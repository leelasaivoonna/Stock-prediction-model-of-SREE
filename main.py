import pickle
from flask import Flask
from werkzeug.wrappers import request, response
from model_files.ml_model import array

app = Flask("mpg_prediction")

@app.route('/', methods = ['POST'])
def predict():
    stock_config = request.get_json()

    with open('SRE_data.csv','rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = array()

    response ={
        'stock_prediction': list(predictions)

    }

    return response
#@app.route('/',methods =['Get'])
#def ping():
    #return "hey"

if __name__ == "__main__":
    app.run(debug=True, host ='0.0.0.0', port=9696)