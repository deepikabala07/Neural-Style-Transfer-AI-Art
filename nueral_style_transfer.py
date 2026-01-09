import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess image
def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img[np.newaxis, :]
    return tf.keras.applications.vgg19.preprocess_input(img)

# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor[0])

# Load images
content_image = load_image("content_images/content.jpg")
style_image = load_image("style_images/style.jpg")

# Load VGG19
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Layers
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1', 'block5_conv1'
]

# Feature extraction model
def vgg_model():
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return tf.keras.Model([vgg.input], outputs)

model = vgg_model()

# Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# Extract features
style_outputs = model(style_image)
content_outputs = model(content_image)

style_features = [gram_matrix(style) for style in style_outputs[:5]]
content_features = content_outputs[5:]

# Initialize generated image
generated_image = tf.Variable(content_image, dtype=tf.float32)

# Optimizer
opt = tf.optimizers.Adam(learning_rate=0.02)

# Style transfer
epochs = 300
style_weight = 1e-2
content_weight = 1e4

for i in range(epochs):
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        style_output_features = outputs[:5]
        content_output_features = outputs[5:]

        style_loss = tf.add_n([
            tf.reduce_mean((gram_matrix(style_output_features[i]) - style_features[i])**2)
            for i in range(5)
        ])

        content_loss = tf.add_n([
            tf.reduce_mean((content_output_features[i] - content_features[i])**2)
            for i in range(len(content_features))
        ])

        loss = style_weight * style_loss + content_weight * content_loss

    grad = tape.gradient(loss, generated_image)
    opt.apply_gradients([(grad, generated_image)])

    if i % 50 == 0:
        print(f"Step {i}, Loss {loss.numpy()}")

# Save output
final_image = tensor_to_image(generated_image.numpy())
final_image.save("output/generated_image.png")
