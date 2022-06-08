
import base64
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL.Image as Image 
import io
import tensorflow as tf 
import numpy as np
import yaml

from tensorflow import keras 
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask_mysqldb import MySQL
from flask import Flask, request, jsonify

app = Flask(__name__)

# database connection
db = yaml.safe_load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app) 

model = keras.models.load_model("IncDogBreed.h5") #load model

def base64conv(img64): #converts base64 
    img64con = base64.b64decode(img64)
    byteImgIO = io.BytesIO()
    img64fin = Image.open(io.BytesIO(img64con))
    img64fin.save(byteImgIO, "png")
    byteImgIO.seek(0)
    img64fin = byteImgIO.read()
    return img64fin



def transform_image(pillow_image):    
    data = []
    img = pillow_image.resize((300,300))
    img = img_to_array(img)
    img = img.astype(np.float32) / 255
    data.append(img)
    data = tf.image.resize(img, [300,300])
    data = np.expand_dims(data, axis=0)
    
    return data

async def predict(x):
    list = ['"beagle" ', '"bull mastiff"', ' "chihuahua" ', ' "german shepherd" ', '"golden retriever"', '"maltese"', '"pomeranian"', ' "pug" ', '"shih tzu"', ' "siberian husky" ']
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = list[np.argmax(pred0)]
    return label0


@app.route("/predict", methods=["POST"])
async def predict_dog():
    if request.method == "POST":
        incoming_jsondata = request.get_json()
        file = incoming_jsondata['data64']

        try:
            file2 = base64conv(file)
            pillow_img = Image.open(io.BytesIO(file2),)
            tensor = transform_image(pillow_img)
            prediction = await predict(tensor)
            #query
            cur = mysql.connection.cursor()
            dog_prediction =cur.execute(f"SELECT * FROM dogs WHERE dogname ={prediction}" )
            if dog_prediction > 0 :
                dogdetail = cur.fetchall()
                return jsonify (dogdetail)
            else:
                return "unknown"
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

@app.route("/<dogid>", methods=["GET"])
def spec(dogid):
    if request.method == "GET":
        cur = mysql.connection.cursor()
        dog_spec = cur.execute(f"SELECT * FROM dogs WHERE dogname = {dogid}"  )
        if dog_spec > 0 :
            dogdetail_id =  cur.fetchall()
        return jsonify(dogdetail_id)

@app.route("/", methods=["GET"])
def alldogs():
    if request.method == "GET":
        cur = mysql.connection.cursor()
    dogs_all = cur.execute("SELECT * FROM dogs")
    if dogs_all > 0 :
        all_dogs =  cur.fetchall()
    return jsonify(all_dogs)

if __name__ == "__main__":
    app.run(debug=True)
