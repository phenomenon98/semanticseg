# SwitchOn Assignment 
<br>
<br>

## Problem Statement
Come up with a Deep Learning based solution that can differentiate between good metal parts and defected metal parts. The model should be able to clearly visualize the location of the defect. The solution should be generalized for a wide range of metallic parts and defects.
<br>
<br>

## My Solution
### Multiclass Classifier + Grad-CAM
I had experience using Deep Learning to build multilabel image classifiers. However, the only visualization method for CNNs that I had heard of was Grad-CAM. Therefore I implemented an image classifier by performing transfer learning with resnext101 pretrained on ImageNet. However, Grad-CAM is useful only to understand what region the network based it's decision off, which did not fit with the problem statement.
<br>
<br>

### Semantic Segmentation
While searching for datasets of metal parts, I stumbled upon [Severstal's Steel Defect Detection Dataset](https://www.kaggle.com/c/severstal-steel-defect-detection). Through this I discovered Semantic Segmentation. I found a library called [segmentation_models](https://github.com/qubvel/segmentation_models) which provides abstracted models of famous CNN architectures often used in Semantic Segmentation tasks. 
<br>
<br>

### Dataset Choice
I wanted a dataset that would generalize well for a wide range of semantic segmentation tasks of machine parts. Severstal's Steel Defect Detection Dataset is a good candidate for this as it contains image of sheet steel with various surface defects that happen during manufacturing.  This is very similar to identifying whether a part has a missing bolt, whether a machine part is rusted, etc., all of which are surface defect detection problems. 
<br>

The dataset contains 12,568 training images and 5,506 test images. The defects are classified into 4 types without any descriptive characteristics. Each image may have one or multiple defects. The defects are marked as masks that are RLE encoded
<br>
<br>

### Model Selection and Pipeline
The models were trained on a NVIDIA Quadro P5000 which allowed me to use much bigger models than what I could use in Colab's default GPU. However, I had limited access to the GPU and could only train each model for a few epochs.
<br>

I trained a binary classifier, a multilabel classifier and a semantic segmentation network individually on each of the 4 types of defects. 
<br>

The binary classifier and the multilabel classifier both used DenseNet121 as the base model. I chose DenseNet121 as it had an above average top1 score on the imagenet dataset while being extremely lightweight. DenseNet121 outperforms several models with similar sizes. Also, larger versions of DenseNet took much longer to train without much increase in f1 score.
<br>

For the Semantic Segmentation network I used an UNet with efficientnetb2 as the backbone. UNet is famous for being very succesfull in semantic segmentation tasks, EfficientNetb2 was the best model I could train due to time constraints. 
<br>
An ensemble model can be designed that uses all these models.
The inference pipeline can follow the following steps:
1. Predict whether part has defect or not. If yes proceed to 2nd step else return no defect and end.
2. Predict what all defects were present in the image. Store the predicted values.
3. Iterate through the predicted defect labels of the image and use the semantic segmentation model to predict the mask region corresponding to the defect. 
<br>
<br>

### Metrics
For classification tasks(Binary and multilabel), I used f1 score as the metric to improve. In order to calculate f1 score I had to calculate precision and recall. These metrics are more meaningful than accuracy, particularly in datasets with high class imbalance.
<br>

Common metrics for semantic segmentation include Pixel accuracy, Intersection over Union and Dice coefficient. Pixel accuracy, though easy to implement, is not a good metric in cases of class imbalance. The dataset is heavily skewed so this was not a good option. IOU and Dice coefficient are highly correlated and very similar to compute. The metric I went with was Dice coefficient.
<br>

Runtime metrics were graphed and observed in realtime using WandB.
<br>
<br>

## Ways to potentially improve model performance
* None of the models were trained until optimum fit due to time. Training the models for more epochs until validation score plateau's will improve their performance.
* Experimenting with other network architectures such as PSPNet or RCNN for the segmentation model.
* Using another model as feature extractor in the classifier. I had to use DenseNet121 due to time and computational constraints.
* Using a different backbone for the segmentation model.
* The dataset was large enough to implement finetuned models instead of replacing the final layer. The models would perform transfer learning much better if their weights were initialized to imagenet weights and the whole network was trained rather than the last few layers. However, this is computationally expensive.
<br>
<br>

## Running the notebook
Running on Colab did not always work for me. Sometimes Colab would crash but then offer better resources(more RAM), after which the notebook would run. Most of the code was written and run using Paperspace's free Nvidia Quadro P5000 machine.
Drive links to models:
<br>
<br>

## Conclusion
The ensemble model pipeline is a good framework that can be generalized for a wide range of semantic segmentation tasks. In the use case of identifying regions of metallic parts that are defected, the models trained on this dataset can be used as the base model to perform transsfer learning with very good generalization.
