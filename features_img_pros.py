import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd

# function to generate laplacian


def laplacian_function(image):
    image = image.resize((600, 400))
    image = np.asarray(image)
    image = image / 255.0
    return cv2.Laplacian(image, cv2.CV_64F)


# storing the directory of images
train_clr_img = "CERTH_ImageBlurDataset/TrainingSet/Undistorted"
train_blr_img_1 = "CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred"
train_blr_img_2 = "CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred"
val_dig_img = "CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet"
val_nat_img = "CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet"

print(" Total Training Images Found ".center(100, "*"))
print("Undistorted Images:", len(os.listdir(train_clr_img)))
print("Naturally-Blurred Images:", len(os.listdir(train_blr_img_1)))
print("Artificially-Blurred Images:", len(os.listdir(train_blr_img_2)))

# processing the images and generating the variance and maximum laplacian for of image
print()
print("Processing the Images in Training Directory...")
clear_img_max_laplacian = []
clear_img_var_laplacian = []
for i in os.listdir(train_clr_img):

    image = Image.open(os.path.join(train_clr_img, i)).convert('L')
    laplacian = laplacian_function(image)

    clear_img_max_laplacian.append(laplacian.max())
    clear_img_var_laplacian.append(laplacian.var())

print("Processing of clear images for training done...")


blurred_img_max_laplacian = []
blurred_img_var_laplacian = []
for i in os.listdir(train_blr_img_1):

    image = Image.open(os.path.join(train_blr_img_1, i)).convert('L')
    laplacian = laplacian_function(image)

    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())

for i in os.listdir(train_blr_img_2):

    image = Image.open(os.path.join(train_blr_img_2, i)).convert('L')
    laplacian = laplacian_function(image)

    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())


print("Processing of blur images for training done...")

print("Saving the data in train.csv file...")

labels = np.append(np.zeros(len(clear_img_max_laplacian)),
                   np.ones(len(blurred_img_max_laplacian)))

laplacian_max = clear_img_max_laplacian + blurred_img_max_laplacian
laplacian_var = clear_img_var_laplacian + blurred_img_var_laplacian

train_data = pd.DataFrame({
    'Laplacian_Max': laplacian_max,
    'Laplacian_Var': laplacian_var,
    'Label': labels
})

train_data = train_data.sample(frac=1).reset_index(drop=True)

train_data.to_csv('train.csv', index=False)

print("Saving the data in train.csv file done...")


print("Processing the images in Validation Directory...")

validation_data1 = pd.read_excel(
    "CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx")
validation_data2 = pd.read_excel(
    "CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx")

validation_data1 = validation_data1.rename(
    {"MyDigital Blur": "Images", "Unnamed: 1": "Labels"}, axis='columns')

validation_data2 = validation_data2.rename(
    {"Image Name": "Images", "Blur Label": "Labels"}, axis='columns')

natural_clear_images = validation_data2.loc[validation_data2["Labels"]
                                            == -1, 'Images'].apply(lambda x: x.strip()+'.jpg').values
digital_clear_images = validation_data1.loc[validation_data1["Labels"]
                                            == -1, 'Images'].apply(lambda x: x.strip()).values

natural_blur_images = validation_data2.loc[validation_data2["Labels"] == 1, 'Images'].apply(
    lambda x: x.strip()+'.jpg').values
digital_blur_images = validation_data1.loc[validation_data1["Labels"] == 1, 'Images'].apply(
    lambda x: x.strip()).values

print()
print("Total Natural Clear Images Found: ", len(natural_clear_images))
print("Total Digital Clear Images Found: ", len(digital_clear_images))
print("Total Natural Blur Images Found: ", len(natural_blur_images))
print("Total Digital Blur Images Found: ", len(digital_blur_images))
print()

val_clear_img_max_laplacian = []
val_clear_img_var_laplacian = []

val_blur_img_max_laplacian = []
val_blur_img_var_laplacian = []

for i in natural_clear_images:
    image = Image.open(os.path.join(val_nat_img, i)).convert('L')
    laplacian = laplacian_function(image)

    val_clear_img_max_laplacian.append(laplacian.max())
    val_clear_img_var_laplacian.append(laplacian.var())

for i in digital_clear_images:
    image = Image.open(os.path.join(val_dig_img, i)).convert('L')
    laplacian = laplacian_function(image)

    val_clear_img_max_laplacian.append(laplacian.max())
    val_clear_img_var_laplacian.append(laplacian.var())


for i in natural_blur_images:
    image = Image.open(os.path.join(val_nat_img, i)).convert('L')
    laplacian = laplacian_function(image)

    val_blur_img_max_laplacian.append(laplacian.max())
    val_blur_img_var_laplacian.append(laplacian.var())

for i in digital_blur_images:
    image = Image.open(os.path.join(val_dig_img, i)).convert('L')
    laplacian = laplacian_function(image)

    val_blur_img_max_laplacian.append(laplacian.max())
    val_blur_img_var_laplacian.append(laplacian.var())


val_laplacian_max = val_clear_img_max_laplacian + val_blur_img_max_laplacian
val_laplacian_var = val_clear_img_var_laplacian + val_blur_img_var_laplacian

print("Saving the validation data in validation.csv file")

labels = np.append(np.zeros(len(val_clear_img_max_laplacian)),
                   np.ones(len(val_blur_img_max_laplacian)))

val_data = pd.DataFrame({
    'Laplacian_Max': val_laplacian_max,
    'Laplacian_Var': val_laplacian_var,
    'Label': labels
})

val_data = val_data.sample(frac=1).reset_index(drop=True)

val_data.to_csv("validation.csv", index=False)

print("Program is executed successfully")
