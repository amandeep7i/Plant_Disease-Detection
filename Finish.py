from image_enhancing import preprocess_image, downscale_image, model,plot_image,psnr
from plant_disease import load_image, predict, predict
import os
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import time

I_disease = input("Enter the path of the Image or the file name in present directory:")
i_PIL = Image.open(I_disease)

shutil.copy("input.jpg","test.jpg")
print("Image selected by the User:")
plot_image(i_PIL)
plt.show()

start = time.time()
hr_image = preprocess_image(I_disease)
lr_image = downscale_image(tf.squeeze(hr_image))
fake_image = model(lr_image)
fake_image = tf.squeeze(fake_image)
plot_image(tf.squeeze(fake_image), title="Super Resolution")
# Calculating PSNR wrt Original Image
psnr = tf.image.psnr(
    tf.clip_by_value(fake_image, 0, 255),
    tf.clip_by_value(hr_image, 0, 255), max_val=255)
print("PSNR Achieved: %f" % psnr)

fig, axes = plt.subplots(1, 3)
plot_image(tf.squeeze(hr_image), title="Original")
plt.subplot(132)
fig.tight_layout()
plot_image(tf.squeeze(fake_image), "Super Resolution")
plt.savefig("output_file.jpg", bbox_inches="tight")
# plt.savefig("saved_file.jpg", bbox_inches="tight")
# Time taken to construct the hr,lr,fake_images for the model and get the output of image _enhancement
print("Time Taken: %f" % (time.time() - start))


# After enhacing the image we have to give input of the output image to the plant_disease detection model for disease detection
new_image = cv2.imread(I_disease)
input_img = load_image(I_disease)
prediction = predict(input_img)
print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
plot_image(tf.squeeze(new_image), title="Original Image")