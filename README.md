# Object Recognition and Detection Challenge - NG PSU Innovation Hack Week

## Project Overview

This repository contains our solution for the **Object Recognition and Detection Challenge** as part of **PSU Fall 2024 Innovation Hack Week**. Our team is tasked with developing a computer vision model that can accurately identify and detect common objects in home and office environments from a curated dataset of images.


## Project Goals

Our primary objectives are:
1. **Object Identification**: Accurately classify the provided images based on the objects they contain.
2. **Object Detection**: Identify and locate objects within images, labeling their bounding boxes.
3. **Efficiency**: Ensure that the model operates efficiently in terms of speed and resource usage.

## Project Approach 
Our approach to the project began with a comprehensive exploration and analysis of the existing models at our disposal. Early on, we recognized that no single model could address all our needs perfectly; each model had its own set of strengths and weaknesses, making it well-suited for specific scenarios but less effective in others. For instance, some models excel at detecting smaller objects but may struggle with large-scale detections, while others perform well in high-contrast images but falter under low-light conditions. Understanding these nuances was a crucial first step in shaping our project strategy.

To gain a thorough understanding, we rigorously tested several models, including various versions and subcategories such as YOLOv8, YOLOv11, and Detectron2. We also evaluated multiple convolutional neural networks (CNNs) like R-CNN, Faster R-CNN, and their respective variations. Each model was assessed for its performance on different types of images, allowing us to identify where they excelled and where they fell short.

Through this extensive testing, it became clear that a single-model approach would not suffice. Instead, we needed a way to leverage the unique strengths of each model while mitigating their weaknesses. This realization led us to develop a method for combining the individual strengths of each model, forming a cohesive and robust ensemble system that could perform effectively across a wide range of detection scenarios. This led us to an Ensembled Model. 


## The Ensemble Model 
Our Ensemble Model strategically combines the strengths of YOLO and Detectron2 to create a powerful and highly accurate object detection system. It leverages a sophisticated pipelined approach, designed to maximize the unique advantages of each individual model while minimizing their respective weaknesses.

In this well-constructed pipeline, images are processed independently by YOLO and Detectron2, with each model performing detections in parallel. Once the models have completed their analysis, their outputs are intelligently aggregated to produce a cohesive and optimized final result. This approach ensures that we harness YOLO's exceptional speed and efficiency alongside Detectron2's precision and ability to handle complex scenes, resulting in a comprehensive and reliable detection system.

Furthermore, the ensemble model has been built with scalability in mind. The flexible design allows for the seamless integration of additional models in the future, which can further enhance performance and improve the aggregation of results. This scalability ensures that our system is adaptable and can continue to evolve as advancements in object detection technology emerge.


## Future Plans 
Despite not having the time to implement all the features we envisioned, we have laid out a solid foundation for a more advanced and robust ensemble model in the future. Our current setup is already designed with scalability and efficiency in mind, and we have taken steps to prepare for future enhancements that will further optimize our model’s performance. This includes adding a framework for multithreading and implementing faster models such as FasterRCNN.



