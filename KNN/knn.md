# K Nearest Neighbor (KNN)

The K Nearest Neighbor (KNN) algorithm is a simple yet powerful classification algorithm. It is a non-parametric algorithm, meaning it does not make any assumptions about the underlying data distribution.

## How KNN works

1. **Step 1: Load the data**: Start by loading the dataset that contains the features and corresponding labels.

2. **Step 2: Choose the value of K**: K is a hyperparameter that determines the number of nearest neighbors to consider for classification. It is important to choose an appropriate value of K, as it can significantly impact the performance of the algorithm.

3. **Step 3: Calculate distances**: For each data point in the dataset, calculate the distance between the input data point and all other data points. The most commonly used distance metric is Euclidean distance.

4. **Step 4: Find the K nearest neighbors**: Select the K data points with the shortest distances to the input data point.

5. **Step 5: Make predictions**: Once the K nearest neighbors are identified, assign the class label that appears most frequently among the K neighbors as the predicted class label for the input data point.

6. **Step 6: Evaluate the model**: Finally, evaluate the performance of the KNN model using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.

## Advantages of KNN

- Simple and easy to understand.
- No training phase, as the algorithm directly uses the training data for classification.
- Can handle multi-class classification problems.
- Can be used for both classification and regression tasks.

## Limitations of KNN

- Computationally expensive, especially for large datasets.
- Sensitive to the choice of K and the distance metric.
- Requires a significant amount of memory to store the entire training dataset.
- Performs poorly when the dataset has irrelevant features or noisy data.