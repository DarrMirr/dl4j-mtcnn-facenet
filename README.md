# dl4j-mtcnn-facenet

**Face recognition application uses:**
- Multi-task Cascaded Convolutional Networks (MTCNN) to detect faces on image
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
- make some optimization to perform face detect on image with large size


### Project requirements

- Java 11+
- Maven 3.6+

### How to run application


1. Open terminal (Linux) or CMD (Windows)
1. Execute command `mvn clean compile` from project's source root directory (You can use absolute path to maven file. For example, `/usr/share/maven/bin/mvn clean compile`)
1. Move to `target` directory that is created by maven at project's source root one
1. Execute command `java -cp "classes/:classes/lib/*" com.github.darrmirr.Application`

Application startup depends on computer's performance and amount of image to train.
Message  `Insert image path or print 'exit' to close:` shows that application is ready.

### How to pass image to test

1. Run application
1. Wait until `Insert image path or print 'exit' to close:` is printed on screen
1. Paste from clipboard absolute path to image file to test
1. Press "Enter"

Log output example:
```
2021-02-08 23:21:50.084  INFO 13507 --- [           main] com.github.darrmirr.Application          : Insert image path or print 'exit' to close:
/home/darrmirr/IdeaProjects/github/dl4j-facenet-mtcnn/src/main/resources/images/dataset/test/Katy_Perry/02-Katy-Perry.jpg
2021-02-08 23:22:04.491  INFO 13507 --- [           main] com.github.darrmirr.FaceDetector         : start : 02-Katy-Perry.jpg
2021-02-08 23:22:10.033  INFO 13507 --- [           main] com.github.darrmirr.FaceDetector         : Extract features from faces : 1
2021-02-08 23:22:13.615  INFO 13507 --- [           main] com.github.darrmirr.FaceDetector         : end : 02-Katy-Perry.jpg
2021-02-08 23:22:13.638  INFO 13507 --- [           main] c.g.d.featurebank.DataSetFeatureBank     : similarity with Katy_Perry is 0.9061704874038696 (min distance)
2021-02-08 23:22:13.638  INFO 13507 --- [           main] com.github.darrmirr.Application          : Insert image path or print 'exit' to close:
```

### How to add new images to train

1. Go to project's source root directory 
1. Go to `src/main/resources/image/dataset/train` directory
1. Create directory with required person name (Directory name is matter due to it is label at application feature bank)
1. Copy images to created directory (Images filename is not matter) 
1. Run application

CAUTION:
- Only one person's face must be on **train** image

### FAQ

1. Is it required to crop images?

Answer: No. Application resize image if it is greater than 600 px at one side. 

2. Is it required that only one person's face should be on **test** image

Answer: No. Application try to find as many faces as possible. Face recognition is performed for all faces on image.