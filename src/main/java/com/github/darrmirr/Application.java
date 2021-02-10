package com.github.darrmirr;

import com.github.darrmirr.featurebank.FeatureBank;
import com.github.darrmirr.utils.ImageFace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationStartedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.FileUrlResource;
import org.springframework.core.io.Resource;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Scanner;

@SpringBootApplication
public class Application {
    private static final Logger logger = LoggerFactory.getLogger(Application.class);
    private Resource[] trainImages;
    private FeatureBank featureBank;
    private FaceDetector faceDetector;

    @Autowired
    public Application(
            @Value("classpath:images/dataset/train/*/*.*") Resource[] trainImages,
            @Qualifier(FeatureBank.DATA_SET) FeatureBank featureBank,
            FaceDetector faceDetector
    ) {
        this.trainImages = trainImages;
        this.featureBank = featureBank;
        this.faceDetector = faceDetector;
    }

    public static void main(String[] args) {
        SpringApplication
                .run(Application.class, args)
                .close();
    }

    @EventListener
    private void onApplicationStartup(ApplicationStartedEvent event) throws IOException {
        logger.info("Filling feature bank : start");
        for (Resource trainImage : trainImages) {
            var faceFeatures = faceDetector.getFaceFeatures(trainImage);
            for (ImageFace imageFace : faceFeatures.getImageFaces()) {
                var label = getLabel(trainImage);
                featureBank.put(label, imageFace.getFeatureVector());
            }
        }
        logger.info("Filling feature bank : end");

        Scanner sc = new Scanner(System.in);
        while(true) {
            logger.info("Insert image path or print 'exit' to close:");
            String inputLine = sc.nextLine();
            if (inputLine.equalsIgnoreCase("exit")) {
                break;
            }
            Resource resource = new FileUrlResource(inputLine);
            var faceFeatures = faceDetector.getFaceFeatures(resource);
            for (ImageFace imageFace : faceFeatures.getImageFaces()) {
                featureBank.getSimilar(imageFace.getFeatureVector());
            }
        }
    }

    private String getLabel(Resource resource) throws IOException {
        return Optional
                .ofNullable(resource)
                .map(Resource::getFilename)
                .map(resource.getURL().getPath()::split)
                .filter(urlParts ->
                        urlParts.length != 0)
                .map(urlParts ->
                        urlParts[0].split(File.separator))
                .filter(urlParts ->
                        urlParts.length != 0)
                .map(urlParts ->
                        urlParts[urlParts.length - 1])
                .orElseThrow();
    }

}
