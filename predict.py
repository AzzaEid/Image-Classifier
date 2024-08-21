# Make all necessary imports.
import argparse
import json
import numpy as np
import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore
from PIL import Image
import os

# Define global variables
IMG_SIZE = 224  
MODEL_PATH = 'my_trained_model.h5' 
CLASS_NAMES_JSON = 'class_names.json' 

# Define the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image /= 255.0
    image = image.numpy()
    return image
    
# Define the predict function
def predict(image_path, model, top_k):

    # Load the image
    image = Image.open(image_path)
    img_array = np.asarray(image)
    img_processed = process_image(img_array)
    
    # Add batch dimension
    expanded_image = np.expand_dims(img_processed, axis=0)
    
    # Make predictions
    predictions = model.predict(expanded_image)
    
    # Get top K predictions
    top_k_values, top_k_indices = tf.math.top_k(predictions[0], k=top_k)
    
    top_probs = top_k_values.numpy().tolist()
    top_classes = top_k_indices.numpy().tolist()
    
    # Convert indices to strings if necessary
    top_classes = [str(cls) for cls in top_classes]  # Adjust if labels are zero-indexed
    
    return top_probs, top_classes


def load_category_names(json_file):
    """
    Load class names from a JSON file.
    
    Args:
    json_file (str): Path to the JSON file mapping class indices to names.

    Returns:
    category_names (dict): Dictionary mapping class indices to names.
    """
    with open(json_file, 'r') as file:
        category_names = json.load(file)
    return category_names

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')
    parser.add_argument('arg1', type=str, help='Path to the image file.')
    parser.add_argument('arg2', type=str, help='Path to the image file.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    parser.add_argument('--category_names', type=str, default=None, help='Path to the JSON file with class names.')
    
    args = parser.parse_args()
    image_path = args.arg1

    # Load the model
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Load class names if provided
    category_names = None
    if args.category_names:
        category_names = load_category_names(args.category_names)
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    """
    Basic usage command:
           python predict.py /path/to/image saved_model
    """
    if args.top_k is None and args.category_names is None:
        probs, classes = predict(image_path, model)
        print("The probabilities and classes of the images: ")

    elif args.top_k is not None:
        top_k = int(args.top_k)
        probs, classes = predict(image_path, model, top_k)
        print("The top {} probabilities and classes of the images: ".format(top_k))
       
    elif args.category_names is not None:
        with open(args.class_names, 'r') as f:
            category_names = json.load(f)
        top_k = int(args.top_k)
        probs, classes = predict(image_path, model,top_k)
        print("The probabilities and classes of the images: ")
        classes = [category_names[class_] for class_ in  classes]
        
            
            
    print("Top {} Predictions:".format(args.top_k))
    for prob, label in zip(probs, classes):
        print(f"{label}: {prob:.4f}")
    
    print('\nThe flower is: "{}"'.format(classes[0]))
    
    
if __name__ == '__main__':
    main()
