<div align="center">

# Image Blur Detection (Predict if an image is blur or clear)

</div>

## Description

This project is modelled using the Laplacian varaince and Laplacian maximum of an image and used to predict if the image is blurred or not. The image with high variance and high maximum are expected to have sharp edges i.e.it's a clear image, whereas the image with less variance and less maximum are expected to be a blur image.

<b>Classifier Used</b>:
K - Nearest Neighbour - 89.2 % Accuracy
Support Vector Machines - 84.59 % Accuracy

## Dependencies and libraries required

Install the given libraries to run the python scripts

<ul>
  <li>numpy==1.19.2</li>
  <li>opencv-python==4.4.0.44</li>
  <li>pandas==1.1.2</li>
  <li>Pillow==7.2.0</li>
  <li>scikit-learn==0.23.2</li>
  <li>pypxl==0.2.3</li>
</ul>

## Dataset Path

<ul>
  <li>CERTH_ImageBlurDataset/TrainingSet/<br>
    <ul>
      <li>Undistorted/</li>
      <li>Naturally-Blurred/</li>
      <li>Artificially-Blurred/</li>
    </ul>
  </li>
  <li>CERTH_ImageBlurDataset/EvaluationSet/ <br>
    <ul>
      <li>DigitalBlurSet/</li>
      <li>NaturalBlurSet/</li>
      <li>DigitalBlurSet.xlsx</li>
      <li>NaturalBlurSet.xlsx</li>
    </ul
  </li>
</ul>

## Data Preprocessing and Feature Engineering

After Completion of the above step, we have to now process the image and perform feature engineering.
Note that this script is run to make 2 new csv files for the test and train data. This step is not necessary as I have already placed the 2 files in the directory. (train.csv, validation.csv). The script also takes some time to run to process the image and generate features.

Execute the below command

```python
python features_img_pros.py
```

## Training the model

After executing the abovr script, we have to now train the model. There are 2 scripts to train the model, one for each algorithm. You can use any for your convenience. Although, K - Nearest Neighbour has the highest accuracy of the two.

<li>Execute the below command (SVM)

```python
python svm_model.py
```

</li>

<li>Execute the below command (KNN)

```python
python knn_model.py
```

</li>

After execution, we can see two files respectively (<b>knnModel.pkl and modelSVM.pkl</b>)

## Test the model

After you have successfully trained either of the models successfully, we can now test to see the accuracy of the model respectively.

<li>Execute the below command (SVM)

```python
python predict_svm.py
```

</li>

<li>Execute the below command (KNN)

```python
python predict_knn.py
```

</li>

## Conclusion / Result

After executing all the steps we can see that we got accuarcy scores as 89 and 85 (rounded) for the 2 models with K - Nearest Neighbour having the highest accuracy of the two.
