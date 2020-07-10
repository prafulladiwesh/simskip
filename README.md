# Project Introduction
This project focused on the Similarity Search of the image and text data using DeepHashing approach using Deep Neural Networks for generating hashcodes for the embedding data and performing the Approximate Nearest Neighbour Search using the Hashed data of the image and text. This is a Scientific Team Project with a team size of 6. 

### Motivation:
Creating hand-crafted features for image data takes a lot of time and manual efforts and it is almost impossible to get data with diverse features. Solution to this problem is to use Embedding data which is very useful in calculating similarity between items.

### System Design:

![Flowchart](https://github.com/prafulladiwesh/simskip/blob/master/Images/designflowchart.png)


### Goal of the Project:
  1. Identify the hyperparametes which has the largest impact on Deep Neural Networks for efficiently learning hash codes.
  2. Check if hashing improves the Top-K similarity search.
  
### Datasets used for this project:
  1. *Image Data:* ImageNet Dataset
      Total images -  1 Million Images
      Images used used for the Experiment - 100,000 (200 classes and 500 images/class)
  2. *Textual Data:* GloVe Dataset

### Implementation Steps:
  1. Generate embedding for the ImageNet dataset using CNN State of the Art architecture EfficientNet.
  2. Word embeddings generated for GloVe dataset.
  3. These embeddings are the input to the DeepHash model.
  4. Trying different Neural Network Architectures with hyperparameter tuning.
  5. Metrics used for evaluations are : Computation and Coverage.
  6. Generate hash codes for all the image and text embeddings.
  7. Perform Top-25 Nearest Neighbour Search on the hash codes as well as the original embeddings.
  8. Evaluate using Recall-25 metric and time report for the search using hashcodes and embeddings.
  
### Results:

![Embed Time](https://github.com/prafulladiwesh/simskip/blob/master/Images/Image_complete_data_time.png)
![Hash Time](https://github.com/prafulladiwesh/simskip/blob/master/Images/Image_hash_data_time.png)

  Top-25 search using DeepHash model takes less time than traditional search using embeddings.

  The Recall-25 value for the image and textual data is almost equal to 1. This is because it was impossible to perform ANN search for the whole dataset due to lack of available hardware. So the Recall-25 using search on bunch of data and taking mean of all the Recall values.
  
### Challenges:
  1. Huge size of ImageHash.csv file (~45GB).
  2. Long training time.
  3. Unavailable hardware.

# Individual Contribution:
  1. Created ANN Search model using Apache Parquet and Spark Server.
  2. Perform Top-K search on image and text data.
  3. Report Recall-25 and Time for the search.
  4. Generate Embeddings using EfficientNet architecture for ImageNet dataset.
  
# How to run the application:

  To perform the ANN search for Top-K images, use : *ImagenetANNSearch.py* which is located at /master/Python%20Files/ImagenetANNSearch.py
  
  To run the file, use the ImageHash.csv file and update the file location in the ImagenetANNSearch.py file.
  
    Command : python3 ImagenetANNSearch.py
