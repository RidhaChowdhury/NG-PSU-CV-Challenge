# rough metrics to get a general idea of how the ensembled model performs 
# does not account for pipeline overhead and image loading - important

import time
import numpy as np
from Models import *
from Ensemble import *
from Aggregators import *

# create a class just to hold the measure of each metric
class Metrics:
    def __init__(self):
        # individual image time
        self.image_times = []
        # yolo computation time
        self.yolo_times = []
        # detr computation time
        self.detr_times = []

    # time to compute each image
    def time_for_each_image(self, time_taken):
        self.image_times.append(time_taken)

    # time taken by each model
    def time_for_each_model(self, model_name, time_taken):
        if model_name == "YoloModel":
            self.yolo_times.append(time_taken)
        elif model_name == "DetrModel":
            self.detr_times.append(time_taken)
        else:
            print("Invalid mode selected \n")

    # print final metrics 
    def final_metric(self):
        total_time = sum(self.image_times)
        average_time = np.mean(self.image_times)
        yolo_total_time = sum(self.yolo_times)
        detr_total_time = sum(self.detr_times)
        yolo_avg_time = yolo_total_time / len(self.image_times) 
        detr_avg_time = detr_total_time / len(self.image_times) 

        print("###################################################################")
        print("Ensemble Model Metrics:")
        print(f"Total processing time for all images: {total_time:.2f} seconds")
        print(f"Average processing time per image: {average_time:.2f} seconds")
        print(f"Total YOLO model inference time: {yolo_total_time:.2f} seconds")
        print(f"Total DETR model inference time: {detr_total_time:.2f} seconds")
        print(f"Average YOLO model inference time per image: {yolo_avg_time:.2f} seconds")
        print(f"Average DETR model inference time per image: {detr_avg_time:.2f} seconds")
        print("###################################################################")




# set the models up
yolo_model = YoloModel()
detr_model = DetrModel()
aggregator = WeightedBoxDiffusion()

# create a collector for the metrics
collector = Metrics()

# call the ensemble pipeline 
class timed_ensemble(ModelInferenceStep):
    # same code from ensemble 
    def __call__(self, image_path, state):
        model_outputs = []
        for model in self.models:
            # start the timer 
            start_time = time.time()
            model_output = model.detect(image_path)
            # detection ends so stop timer
            end_time = time.time()
            model_name = model.__class__.__name__
            # add to collector
            collector.time_for_each_model(model_name, end_time - start_time)
            model_outputs.append((model_output["boxes"], model_output["scores"], model_output["classes"], model_name))

        return model_outputs, state

model_inference_step = timed_ensemble(models=[yolo_model, detr_model])
model_inference_render_step = ModelInferenceRenderStep()
aggregation_step = AggregationStep(aggregator=aggregator)
render_step = RenderStep(class_names={0: "person", 1: "car"})

ensemble_pipeline = EnsemblePipeline([
    model_inference_step,
    model_inference_render_step,
    aggregation_step,
    render_step
])

# Process all images and log performance
image_directory = './data/Images/'
image_filenames = [f"image{i}.jpg" for i in range(1, 21)]
start_time_total = time.time()

# loop through each image 
for image_filename in image_filenames:
    image_path = image_directory + image_filename
    input_state = {'image_path': image_path}

    start_time_image = time.time()
    output = ensemble_pipeline.process(image_path, input_state)
    end_time_image = time.time()
    
    # add the time for each image
    collector.time_for_each_image(end_time_image - start_time_image)
    print(f"Processed {image_filename} in {end_time_image - start_time_image:.2f} seconds")

end_time_total = time.time()

# print the final metrics here
collector.final_metric()
