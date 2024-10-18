# README.md

## Instructions for Running the Jupyter Notebook

The primary Jupyter/Google Colab notebook for this project is `final_pipeline.ipynb`. It is divided into six sections:

1. **Library Installation and Model Loading:** Ensure relevant libraries are installed, and models are loaded. Dependencies are selected for compatibility with the Google Colab environment.
2. **Model Upload:** Upload the models directly to the `/content/` directory and provide the correct paths in the third cell.
3. **Initialization:** Run all cells once at the beginning to initialize the necessary variables and functions.
4. **Inference and Digitization Functions:** The last two sections contain the inference and digitization functions, which can be called by passing the image location as a parameter.

## Abstract

Extracting medical vitals from images is crucial yet time-consuming. With the increasing efficacy of Artificial Intelligence (AI), many are working to automate healthcare activities, including medical entity extraction and X-ray processing. This work proposes a transfer learning-based object segmentation model for vital extraction, which takes an image of a monitor as input and outputs various medical vitals, such as oxygen levels and heart rates. The model first identifies the monitor segment, marks the region of interest for vitals, and then recognizes the marked boxes along with their respective vital labels. Our proposed pipeline approach helps locate errors effectively and allows for easy scalability. Our model achieves reliable identification of different vitals with confidence levels ranging from 0.9 to 0.97. A thorough human evaluation shows the model's robustness in handling uncertainties such as monitor angle, shape, and background color.

## Architecture

### Overview
The model emulates human vision in reading vitals. It first locates the monitor, determines its type, and identifies the digits based on their position, size, and color. The model proceeds by detecting the monitor screen, classifying it using a cropped image, and labeling the data, which is then passed to the OCR for reading and reporting results.

### Monitor Segmenter
We used YOLOv5 (You Only Look Once), a state-of-the-art model for quick object detection, to identify the monitor in an image. The model provides the x-center, y-center, height, and width of the detected monitor. We select the monitor with the highest confidence score, crop it, and pass it to later stages.

![Monitor Detection Example](image_folder/image_page2_0.png)
![Monitor Detection Visualization](image_folder/image_page2_1.png)

### Monitor Classifier
After obtaining the bounding box of the monitor, we classify it into four classes using K-means clustering with the Inception V3 model, a CNN-based deep learning architecture. This approach, documented in the `image_feature` function in the final pipeline, produces the monitor label as output, introducing novelty into our pipeline through unsupervised learning.

### Digit Segmenter
To detect digits in monitor images, we again utilized the YOLOv5 system, known for its speed and accuracy. The YOLOv5 model (loaded as `digit_segmenter` in the Colab notebook) outputs the x-center, y-center, height, and width of detected digits. To minimize false positives, we select the top six digits based on confidence scores for further processing.

### Vital Classification
The digit location and size information are fed into an XGBoost classifier to predict the vital type. XGBoost is a flexible and powerful ML model that uses decision trees as its base learner and improves predictions through gradient boosting. The final results are returned as a dictionary object populated with key-value pairs obtained from OCR.

### OCR
We employ EasyOCR, which utilizes deep learning algorithms for efficient text recognition. The results are returned in a dictionary format. Before passing images to OCR, we enhance sharpness and adjust brightness to mitigate glare.

### Heart Rate Digitization
A separate YOLOv5 model was trained on the complete classification dataset to extract heart rate graph coordinates. We crop the graph image and apply contour detection via OpenCV to extract the coordinates, avoiding edge detection due to its limitations in handling noise. The detected coordinates are plotted using Matplotlib.

## Training and Other Details

### Monitor Segmenter
The YOLOv5 model was trained on 1800 of the 2000 image samples and tested on the remaining 200. Training was conducted for 25 epochs with a batch size of 8. The model's efficacy is summarized as follows:

| Metric     | Value   |
|------------|---------|
| Precision  | 0.99    |
| Recall     | 1       |
| mAP50      | 0.995   |
| mAP50-95   | 0.99    |

### Monitor Classifier
We initially utilized Microsoft’s ResNet-18 for monitor classification. However, we shifted to the Inception V3 model, focusing on monitor layout instead of type, which proved more effective for unsupervised learning. This approach resulted in 100% accuracy on the training and testing datasets of 1000 monitor classification images.

### Digit Segmenter
To achieve a machine-agnostic model, we trained it on all four classification datasets, treating all vitals as a single class. The model was trained with a batch size of 10 for 30 epochs. We faced challenges with false positives labeling the background as digits. To address this, we only considered the top six scores with the highest confidence for digit classification.

#### Validation Parameters
The following table summarizes the validation metrics for digit detection:

| Metric                 | Value    |
|------------------------|----------|
| Recall                 | 0.99627  |
| mAP_0.5                | 0.9942   |
| mAP_0.5:0.95           | 0.80342  |
| Box Loss               | 0.015804 |

### Vital Classification
An XGBoost model was specifically trained for each of the four monitor classes to accurately classify digits. The position and size of the digit's bounding box served as input features. Each class was trained on approximately 250x6 samples, with hyperparameter optimization conducted through grid search for improved accuracy.

| Monitor Class                               | Accuracy |
|---------------------------------------------|----------|
| BPL-Ultima-PrimeD-A-classification         | 100%     |
| BPL-EliteView-EV10-B_Meditec-England-A     | 98.69%   |
| BPL-EliteView-EV100-C                       | 97.0%    |
| Nihon-Kohden-lifescope                      | 99%      |

### OCR Performance
We found that EasyOCR effectively handled the challenge of detecting a mix of alphanumeric characters. It allows specifying certain characters for recognition, leading to improved accuracy. We tested EasyOCR, Pyterreact, and Calamari OCR, with EasyOCR demonstrating the best performance. Additionally, we enhanced image sharpness and brightness before OCR processing.

### Heart Rate Digitization Process
To align graph peaks, we rotated the cropped image based on the mean angles formed by lines connecting the mean line to peaks and troughs. By isolating noise with the largest continuous contour, we obtained reliable graph coordinates. Multiple y-coordinates for a single x-coordinate were consolidated using the mean.

## Conclusion
This study addresses the challenge of accurately recognizing and digitizing heart rate and other vitals from various monitoring devices in low-resource settings. We demonstrate that open-source models and data can achieve satisfactory performance on the trained dataset. Future work may focus on developing zero-shot or few-shot solutions to enhance this process.
