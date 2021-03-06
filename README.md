# Image Classification
A research on image classification and auto insurance claim prediction, a systematic experiments on modeling techniques and approaches. The image data was extracted from Kaggle. 

This project is to predict the condition of cars (whether they are damaged or not) based on images and the amount of insurance claims of the cars if they are damaged based on relevant information provided. This research explores a problem that contains image and tabular data as inputs, which better simulates real-world problems that are likely to integrate structured, semi-structured and unstructured data as inputs to perform predictive analysis and generate actionable insights. 

## How it works
First, the image data was used as input for the neural network (NN) classifier. This is to predict and classify the condition of cars as either 0, not damaged or 1, damaged. The NN classifier is pre-trained using ImageNet data as default weights to detect images. The pre-trained NN classifier is then used for this model to classify the condition through transfer learning. Consequently, the data is merged with tabular data to undergo several modelling algorithms for tabular analysis. Using the results of image classification, different algorithms can now be used to predict the insurance claim amount for those cars that are damaged. Method 1 involves the conventional modeling techniques such as the four models constructed in this study: Decision Tree, Linear Regression, Gaussian Naïve Bayes and K-Nearest Neighbors (KNN) while Method 2 is AutoML. With this pipeline, unstructured data (image data) and structured data (tabular data) can be supported for predictive analysis by streaming the data through the pipeline for a complete predictive analysis.

## Conclusion 
AutoML models are able to fit the tabular data to predict the insured amount better than other conventional methods including linear regression, decision tree, Gaussian Naïve Bayes and KNN. AutoML also provides an advantage over the conventional models as it is able to take care of the data preprocessing, feature engineering and model generation automatically. For instance, the powerful feature of AutoML to test out multiple data preparation techniques in a single run makes it highly suitable to become the basis to build machine learning models in the near future as machine learning predictions become more ubiquitous in daily applications. 

## Data Source
Kaggle: [Fast, Furious and Insured](https://www.kaggle.com/infernape/fast-furious-and-insured)

## Contributors
This is a group project completed in collaboration with: <br>
<a href="https://github.com/MagdaleneHo/Image-Classification/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MagdaleneHo/Image-Classification" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
