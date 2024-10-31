import numpy as np
import time

class EnsemblePipeline:
    def __init__(self, pipeline_steps):
        """
        Initializes the ensemble pipeline with a list of callable steps.
        
        Args:
            pipeline_steps (list): List of callables where each step processes 
                                   the data and passes it to the next step.
        """
        self.pipeline_steps = pipeline_steps

    def process(self, data, state=None):
        """
        Passes data through each step in the pipeline.
        
        Args:
            data: Initial input data (e.g., image path or pre-processed data).
        
        Returns:
            The final output after passing through all pipeline steps.
        """
        if state is None:
            state={}

        for step in self.pipeline_steps:
            print("\033[94mStep:", step.__class__.__name__, "\033[0m")
            data, state = step(data, state)
            print("\033[92mOutput:", data, "\033[0m")
            print("\033[92mState:", state, "\033[0m")
        return data, state


class ModelInferenceStep:
    def __init__(self, models):
        """
        Initializes the model inference step with a list of models.
        
        Args:
            models (list): List of Model instances to use for inference.
        """
        self.models = models

    def __call__(self, image_path, state):
        """
        Runs inference on all models for the given image path.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            list: List of model outputs in the format (boxes, scores, classes).
        """
        model_outputs = []
        for model in self.models:
            model_output = model.detect(image_path)
            model_outputs.append((model_output["boxes"], model_output["scores"], model_output["classes"], model.__class__.__name__))
        return model_outputs, state
    

import os
class ModelInferenceRenderStep:
    def __call__(self, model_outputs, state):
        """
        Renders the aggregated bounding boxes, scores, and class labels on the image.
        
        Args:
            aggregated_output (dict): Aggregated bounding boxes, scores, and classes with image path.
        
        Returns:
            str: Path to the saved output image with rendered boxes.
        """
        image_path = state['image_path']
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the base name without extension
        output_dir = "data/outputted_images"


        for i, (boxes, scores, classes, model_name) in enumerate(model_outputs):
            # render the individual models output
            prediction = {"boxes": boxes, "scores": scores, "classes": classes}
            output_path = output_dir + '/' + base_name + '_' + model_name + '.jpg'
            output_image = RenderStep()(prediction, state)[0]
            print(output_image)
            
            # rename the image output, overwrite if exists
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(output_image, output_path)
            print(f"Image saved to {output_path}")
        return model_outputs, state

class AggregationStep:
    def __init__(self, aggregator):
        """
        Initializes the aggregation step with an aggregator.
        
        Args:
            aggregator (Aggregator): An instance of an aggregation strategy.
        """
        self.aggregator = aggregator

    def __call__(self, model_outputs, state):
        """
        Aggregates model outputs.
        
        Args:
            model_outputs (list): List of model outputs in the format (boxes, scores, classes).
        
        Returns:
            dict: Aggregated bounding boxes, scores, and classes.
        """
        return self.aggregator.aggregate(model_outputs), state


# Example of a custom post-processing step (optional)
class PostProcessingStep:
    def __call__(self, aggregated_output, state):
        """
        A custom post-processing step to modify aggregated results if needed.
        
        Args:
            aggregated_output (dict): Aggregated bounding boxes, scores, and classes.
        
        Returns:
            dict: Post-processed aggregated output.
        """
        # Example: Filter out low-confidence detections (adjust threshold as needed)
        confidence_threshold = 0.5
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []
        
        for box, score, cls in zip(aggregated_output["boxes"], aggregated_output["scores"], aggregated_output["classes"]):
            if score >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_classes.append(cls)
        output = {
            "boxes": np.array(filtered_boxes),
            "scores": np.array(filtered_scores),
            "classes": np.array(filtered_classes),
        }
        return output, state
    
import cv2
import numpy as np
from tqdm import tqdm

class RenderStep:
    def __init__(self, class_names={0: "Class0"}, confidence_threshold=0.5):
        """
        Initializes the rendering step with enhanced annotation options.
        
        Args:
            class_names (dict, optional): Dictionary mapping class indices to class names.
            confidence_threshold (float): Minimum confidence score to display a box.
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

    def __call__(self, aggregated_output, state):
        """
        Renders the aggregated bounding boxes, scores, and class labels on the image with improved quality.
        
        Args:
            aggregated_output (dict): Aggregated bounding boxes, scores, and classes with image path.
        
        Returns:
            str: Path to the saved output image with rendered boxes.
        """
        image_path = state['image_path']
        boxes = aggregated_output["boxes"]
        scores = aggregated_output["scores"]
        classes = aggregated_output["classes"]

        print("Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        print("Image loaded successfully.")

        # Set font scale and thickness based on image size for better visibility
        height, width = image.shape[:2]
        font_scale = max(0.5, min(width, height) / 1250)  # Scale text based on image size
        box_thickness = max(2, min(width, height) // 300)  # Dynamic thickness for bounding boxes

        for i, box in tqdm(enumerate(boxes), total=len(boxes), desc="Processing bounding boxes"):
            score = scores[i]

            # Draw bounding box
            x_min, y_min, x_max, y_max = map(int, box)
            cls = classes[i]
            color = (0, 255, 0)  # Green for bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, box_thickness, cv2.LINE_AA)

            # Label with class name and confidence
            label = f"{self.class_names.get(cls, cls)}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y_min, label_size[1] + 10)

            # Draw label background and text
            cv2.rectangle(image, (x_min, y_min - label_size[1] - 10), 
                          (x_min + label_size[0], y_min), color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x_min, label_y - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Generate output path based on input image name
        output_dir = "./data/outputted_images"
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = output_dir + '/' + base_name + "_aggregated_image.jpg"
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")
        return output_path, state

from Models import YoloModel, DetrModel
from Aggregators import WeightedBoxDiffusion

def run_ensemble(models, folder):

    model_dict = {
    "YoloModel": YoloModel,
    "FasterRCNNModel": "FrcnnModel",
    "DetrModel": DetrModel
    }
    
    # Instantiate models based on input
    instantiated_models = []
    for model_name in models:
        if model_name in model_dict:
            model_class = model_dict[model_name]
            instantiated_models.append(model_class())
        else:
            print(f"Model {model_name} not recognized.")

    aggregator = WeightedBoxDiffusion()
    
    # Define pipeline steps
    model_inference_step = ModelInferenceStep(instantiated_models)
    model_inference_render_step = ModelInferenceRenderStep()
    aggregation_step = AggregationStep(aggregator=aggregator)
    post_processing_step = PostProcessingStep()
    render_step = RenderStep(class_names={0: "person", 1: "car"})  # Example class mapping

    # Create the ensemble pipeline with the rendering step
    ensemble_pipeline = EnsemblePipeline([
        model_inference_step,
        model_inference_render_step,
        aggregation_step,
        render_step
    ])

    # Define acceptable image extensions
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    # Process each image in the folder
    for image_name in os.listdir(folder):
        if image_name.lower().endswith(valid_extensions):
            image_path = os.path.join(folder, image_name)
            input_state = {'image_path': image_path}
            start_time = time.time()
            output = ensemble_pipeline.process(input_state["image_path"], input_state)
            print(f"Processed {image_name} in {time.time() - start_time:.2f} seconds")
        else:
            print(f"Skipping non-image file: {image_name}")