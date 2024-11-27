import tensorflow as tf


UPLOAD_FOLDER = "app/static/uploads/"
STATIC_FOLDER = "app/static"
IMAGE_SIZE = (180, 180)
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "cat_dog.keras")
