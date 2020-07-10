# Introduction of the Project
This project focused on the Similarity Search of the image and text data using DeepHashing approach using Deep Neural Networks for generating hashcodes for the embedding data and performing the Approximate Nearest Neighbour Search using the Hashed data of the image and text.

### Motiovation:
Creating hand-crafted features for image data takes a lot of time and manual efforts and it is almost impossible to get data with diverse features. Solution to this problem is to use Embedding data which is very useful in calculating similarity between items.

### System Design:






### Goal of the Project:
  1. Identify the hyperparametes which has the largest impact on Deep Neural Networks for efficiently learning hash codes.
  2. Check if hashing improves the Top-K similarity search.
  
### Datasets used for this project:
  1. Image Data : ImageNet Dataset
      Total images -  1 Million Images
      Images used used for the Experiment - 100,000 (200 classes and 500 images/class)
  2. Textual Data: GloVe Dataset

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
  
  
