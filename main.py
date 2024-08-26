import tensorflow as tf
import numpy as np
import cv2
#from minio import Minio
import io
import os
import imghdr
from typing import Dict
import matplotlib.pyplot as plt

import rembg
import fast_colorthief
from colorthief import ColorThief
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import webcolors

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import sys

sequential_model = tf.keras.models.load_model('sequentialModel.h5')
mobilenet2_model = tf.keras.models.load_model('mobileNetV2Model.keras')
vgg16_model = tf.keras.models.load_model('vgg16Model.keras')

# Clasificarea imaginilor folosind modelele
class_names = ['pants', 't-shirt', 'skirt', 'dress', 'shorts', 'shoes', 'hat' , 'longsleeve' , 'outwear' ,'shirt']

threshold = 0.6

def prediction(data):
    vgg16_predictions = vgg16_model.predict(data)
    mobilenet2_predictions = mobilenet2_model.predict(data)
    sequential_predictions = sequential_model.predict(data)
    probabilities = vgg16_predictions + mobilenet2_predictions + sequential_predictions  
   

    mask = probabilities > threshold 
    filtered_probabilities = probabilities[mask]  
    filtered_classes = np.arange(len(class_names))[mask.reshape(-1)]  
    print(filtered_classes)
    if len(filtered_classes) > 0:
        max_index = np.argmax(filtered_probabilities)
        predicted_class_index = filtered_classes[max_index]
        predicted_class = class_names[predicted_class_index]
        print('The object in the image is a', predicted_class)
        return predicted_class
    else:
        print('No object detected above the probability threshold')
        return
    
def resize_image(image_bytes, max_size):
    image_format = imghdr.what(None, h=image_bytes)  # Determine the image format
    print(image_format)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!")
    if image_format not in ['png', 'jpeg', 'jpg', 'bmp']:
        raise ValueError(f"Unsupported image format: {image_format}. Please use 'png', 'jpeg', 'jpg', or 'bmp'.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("The input image could not be decoded. Please check the image format and content.")

    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width, new_height = int(scale * width), int(scale * height)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    if image_format == 'jpg':
        image_format = 'jpeg'

    _, buffer = cv2.imencode(f'.{image_format}', resized_image)
    return io.BytesIO(buffer)

def read_image(image: bytes) -> np.ndarray:
    resized_image_bytes = resize_image(image, max_size)
    image = Image.open(resized_image_bytes)
    img = image
    img = np.array(img)  # Convert the PIL.Image object to a NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data = np.array([img]) / 255.0
    return data

# Funcție ce extrage culoarea predominantă
async def extract_dominant_color(img):
    image_bytes = await img.read()
    palette = fast_colorthief.get_palette(io.BytesIO(image_bytes), quality=1)
    print(palette[0])
    dominant_color = palette[0]
    return dominant_color

# Funcție ce calculează distanța euclidiană dintre 2 culori RGB
def euclidean_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

# Funcție ce găsește culoarea cea mai apropiată
def find_closest_color(target_color, color_dict):
    closest_color = None
    min_distance = float('inf')
    for color_name, color_value in color_dict.items():
        distance = euclidean_distance(target_color, color_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color

# Funcție ce returnează numele culorii celei mai apropiate din webcolors
def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


model_name = "u2net_cloth_seg"
app = FastAPI()
max_size = 1000

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000",
                   "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message" : "Hello World!!!"}
logging.info('Method is called')

@app.post("/remove_background/")
async def remove_background(image_file: UploadFile = File(...)) -> StreamingResponse:
    # Read the image file contents
    input_image_bytes = await image_file.read()
    
    # Open image using PIL
    input_image = Image.open(io.BytesIO(input_image_bytes))
    
    # Resize image to 224x224
    resized_image = input_image.resize((224, 224))
    
    # Convert PIL image back to bytes
    resized_image_bytes = io.BytesIO()
    resized_image.save(resized_image_bytes, format='PNG')
    resized_image_bytes = resized_image_bytes.getvalue()

    # Use a different model
    # model_name = "u2net_cloth_seg"
    # session = rembg.new_session(model_name)


    # Remove background
    output_image_bytes = rembg.remove(resized_image_bytes)

    # Save the image
    filename = "output_image.png"
    with open(filename, "wb") as f:
        f.write(output_image_bytes)

    return StreamingResponse(io.BytesIO(output_image_bytes), media_type="image/png")

@app.post("/classify/")
async def classify_image(image_file: UploadFile = File(...)) -> str:
    # Read the image file contents
    image_bytes = await image_file.read()
    
    # Read image 
    img_array = read_image(image_bytes)

    # Call your prediction function
    predictions = prediction(img_array)
    
    if predictions:
        return predictions
    else:
        return 'empty'
    

@app.post("/classify-color/")
async def classify_color(image_file: UploadFile = File(...)) -> str:
    try:
        # Read the image file contents - that is in extract dominant color
        # image_bytes = await image_file.read()
        
        #   # Extract dominant color
        # img = read_image(image_bytes)
        dominant_color = await extract_dominant_color(image_file)
        print("Dominant color is: ", dominant_color)

        # Predefined color dictionary {color_name: (R, G, B)}
        color_dict = {
            'aliceblue': 'white',
            'antiquewhite': 'white',
            'aqua': 'cyan',
            'aquamarine': 'cyan',
            'azure': 'white',
            'beige': 'white',
            'bisque': 'white',
            'black': 'black',
            'blanchedalmond': 'brown',
            'blue': 'blue',
            'blueviolet': 'purple',
            'brown': 'brown',
            'burlywood': 'brown',
            'cadetblue': 'cyan',
            'chartreuse': 'green',
            'chocolate': 'brown',
            'coral': 'orange',
            'cornflowerblue': 'blue',
            'cornsilk': 'brown',
            'crimson': 'red',
            'cyan': 'cyan',
            'darkblue': 'blue',
            'darkcyan': 'cyan',
            'darkgoldenrod': 'brown',
            'darkgray': 'gray',
            'darkgrey': 'gray',
            'darkgreen': 'green',
            'darkkhaki': 'yellow',
            'darkmagenta': 'purple',
            'darkolivegreen': 'green',
            'darkorange': 'orange',
            'darkorchid': 'purple',
            'darkred': 'red',
            'darksalmon': 'pink',
            'darkseagreen': 'green',
            'darkslateblue': 'purple',
            'darkslategray': 'black',
            'darkslategrey': 'black',
            'darkturquoise': 'cyan',
            'darkviolet': 'purple',
            'deeppink': 'pink',
            'deepskyblue': 'blue',
            'dimgray': 'gray',
            'dimgrey': 'gray',
            'dodgerblue': 'blue',
            'firebrick': 'red',
            'floralwhite': 'white',
            'forestgreen': 'green',
            'fuchsia': 'purple',
            'gainsboro': 'white',
            'ghostwhite': 'white',
            'gold': 'yellow',
            'goldenrod': 'brown',
            'gray': 'gray',
            'grey': 'gray',
            'green': 'green',
            'greenyellow': 'green',
            'honeydew': 'white',
            'hotpink': 'pink',
            'indianred': 'red',
            'indigo': 'purple',
            'ivory': 'white',
            'khaki': 'yellow',
            'lavender': 'purple',
            'lavenderblush': 'white',
            'lawngreen': 'green',
            'lemonchiffon': 'yellow',
            'lightblue': 'blue',
            'lightcoral': 'red',
            'lightcyan': 'cyan',
            'lightgoldenrodyellow': 'yellow',
            'lightgray': 'gray',
            'lightgrey': 'white',
            'lightgreen': 'green',
            'lightpink': 'pink',
            'lightsalmon': 'red',
            'lightseagreen': 'cyan',
            'lightskyblue': 'blue',
            'lightslategray': 'gray',
            'lightslategrey': 'gray',
            'lightsteelblue': 'blue',
            'lightyellow': 'yellow',
            'lime': 'green',
            'limegreen': 'green',
            'linen': 'white',
            'magenta': 'purple',
            'maroon': 'brown',
            'mediumaquamarine': 'green',
            'mediumblue': 'blue',
            'mediumorchid': 'pruple',
            'mediumpurple': 'purple',
            'mediumseagreen': 'green',
            'mediumslateblue': 'purple',
            'mediumspringgreen': 'green',
            'mediumturquoise': 'cyan',
            'mediumvioletred': 'pink',
            'midnightblue': 'blue',
            'mintcream': 'white',
            'mistyrose': 'white',
            'moccasin': 'yellow',
            'navajowhite': 'brown',
            'navy': 'blue',
            'oldlace': 'white',
            'olive': 'green',
            'olivedrab': 'green',
            'orange': 'orange',
            'orangered': 'orange',
            'orchid': 'purple',
            'palegoldenrod': 'yellow',
            'palegreen': 'green',
            'paleturquoise': 'cyan',
            'palevioletred': 'pink',
            'papayawhip': 'yellow',
            'peachpuff': 'yellow',
            'peru': 'brown',
            'pink': 'pink',
            'plum': 'purple',
            'powderblue': 'blue',
            'purple': 'purple',
            'rebeccapurple': 'purple',
            'red': 'red',
            'rosybrown': 'brown',
            'royalblue': 'blue',
            'saddlebrown': 'brown',
            'salmon': 'red',
            'sandybrown': 'brown',
            'seagreen': 'green',
            'seashell': 'white',
            'sienna': 'brown',
            'silver': 'gray',
            'skyblue': 'blue',
            'slateblue': 'purple',
            'slategray': 'gray',
            'slategrey': 'gray',
            'snow': 'white',
            'springgreen': 'green',
            'steelblue': 'blue',
            'tan': 'brown',
            'teal': 'cyan',
            'thistle': 'purple',
            'tomato': 'orange',
            'turquoise': 'cyan',
            'violet': 'purple',
            'wheat': 'brown',
            'white': 'white',
            'whitesmoke': 'white',
            'yellow': 'yellow',
            'yellowgreen': 'green',

        }

        # Find the closest color from the predefined dictionary
        closest_color = get_colour_name(dominant_color)
        print("Closest color is: ", closest_color)
        specific_color = color_dict.get(closest_color)
        
        return specific_color
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)