import tensorflow as tf

from app.config import IMAGE_SIZE


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def load_and_preprocess_image(path: str) -> tf.Tensor:
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=IMAGE_SIZE
    )

    return preprocess_image(image)


def classify(model: tf.keras.Model, image_path: str) -> tuple[str, float]:
    preprocessed_image = load_and_preprocess_image(image_path)

    predictions = model.predict(preprocessed_image)
    score = predictions[0][0]

    label = "cat" if score <= 0.5 else "dog"
    prob = 1 - score if label == "cat" else score

    return label, prob
