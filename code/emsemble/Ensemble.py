import numpy as np

class EnsemblePipeline:
    def __init__(self, pipeline_steps):
        """
        Initializes the ensemble pipeline with a list of callable steps.
        
        Args:
            pipeline_steps (list): List of callables where each step processes 
                                   the data and passes it to the next step.
        """
        self.pipeline_steps = pipeline_steps

    def process(self, data):
        """
        Passes data through each step in the pipeline.
        
        Args:
            data: Initial input data (e.g., image path or pre-processed data).
        
        Returns:
            The final output after passing through all pipeline steps.
        """
        for step in self.pipeline_steps:
            data = step(data)
        return data


class ModelInferenceStep:
    def __init__(self, models):
        """
        Initializes the model inference step with a list of models.
        
        Args:
            models (list): List of Model instances to use for inference.
        """
        self.models = models

    def __call__(self, image_path):
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
            model_outputs.append((model_output["boxes"], model_output["scores"], model_output["classes"]))
        return model_outputs


class AggregationStep:
    def __init__(self, aggregator):
        """
        Initializes the aggregation step with an aggregator.
        
        Args:
            aggregator (Aggregator): An instance of an aggregation strategy.
        """
        self.aggregator = aggregator

    def __call__(self, model_outputs):
        """
        Aggregates model outputs.
        
        Args:
            model_outputs (list): List of model outputs in the format (boxes, scores, classes).
        
        Returns:
            dict: Aggregated bounding boxes, scores, and classes.
        """
        return self.aggregator.aggregate(model_outputs)


# Example of a custom post-processing step (optional)
class PostProcessingStep:
    def __call__(self, aggregated_output):
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
        
        return {
            "boxes": np.array(filtered_boxes),
            "scores": np.array(filtered_scores),
            "classes": np.array(filtered_classes),
        }

from Models import YoloModel, DetrModel
from Aggregators import WeightedBoxDiffusion

if __name__ == "__main__":
    # Instantiate models
    yolo_model = YoloModel()
    detr_model = DetrModel()

    # Instantiate aggregator
    aggregator = WeightedBoxDiffusion()

    # Define pipeline steps
    model_inference_step = ModelInferenceStep(models=[yolo_model, detr_model])
    aggregation_step = AggregationStep(aggregator=aggregator)
    post_processing_step = PostProcessingStep()  # Optional

    # Create the ensemble pipeline with the desired steps
    ensemble_pipeline = EnsemblePipeline([
        model_inference_step,
        aggregation_step,
        post_processing_step
    ])

    # Process a single image
    image_path = '../data/Images/image5.jpg'
    output = ensemble_pipeline.process(image_path)
    print("Final Output:", output)
