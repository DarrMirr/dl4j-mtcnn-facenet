# dl4j-facenet-mtcnn

**Face recognition application uses:**
- Multi-task Cascaded Convolutional Networks (MTCNN) to detect face on image
- Inception ResNet V1 neural network to build face feature vector
- Euclidean distance (as default) to calculate similarity between two face feature vectors. 
There is cosine distance verifier in application. Change `@Qualifier(FeatureVerifier.EUCLIDEAN_DISTANCE)` to `@Qualifier(FeatureVerifier.COSINE_DISTANCE)` in constructor of DataSetFeatureBank class in order to switch between two algorithm.

**How to start:**
To start application run main method in Application class

This application build on https://github.com/cfg1234/dl4j-facenet

Main differences:
- make refactoring in order to simplify code
- rewrite mtcnn utils methods using ND4J and DL4J features
- add methods documentation 