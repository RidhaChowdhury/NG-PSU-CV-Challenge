from abc import ABC, abstractmethod
import numpy as np
import torch
from torchvision.ops import box_iou
class Aggregator():
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, predictions):
        pass

class WeightedBoxDiffusion(Aggregator):
    def __init__(self):
        pass

    def aggregate(self, predictions, iou_threshold=0.5):
        # Unpack the model outputs into lists of boxes, scores, and classes
        boxes_list, scores_list, classes_list, model_names = zip(*predictions)

        # Stack all boxes, concatenate all scores and classes
        all_boxes = np.vstack(boxes_list)
        all_scores = np.concatenate(scores_list)
        all_classes = np.concatenate(classes_list)

        # Convert all boxes to a torch tensor for box_iou compatibility
        all_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)

        # Calculate IoU matrix for all pairs of boxes
        iou_matrix = box_iou(all_boxes_tensor, all_boxes_tensor)

        # Print the IoU matrix for debugging
        print("IoU Matrix:")
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                print(f"IoU[{i}][{j}] = {iou_matrix[i, j]:.4f}", end="\t")
            print()  # Newline for readability

        # Lists to store final aggregated results
        final_boxes, final_scores, final_classes = [], [], []
        used_indices = set()

        # Loop through each box and aggregate based on IoU only
        for i, (box, score, cls) in enumerate(zip(all_boxes, all_scores, all_classes)):
            if i in used_indices:
                continue

            # Find all boxes that overlap with the current box, ignoring class
            overlaps = (iou_matrix[i] > iou_threshold).nonzero(as_tuple=False).flatten()
            group_indices = [j.item() for j in overlaps if j.item() not in used_indices]

            # Debug: Print group information for analysis
            print(f"\nBox {i}: Initial box {box}, Score: {score}, Class: {cls}")
            print(f"Group indices (overlapping boxes with IoU > {iou_threshold}): {group_indices}")
            print(f"IoUs with group indices: {[iou_matrix[i][j].item() for j in group_indices]}")

            if not group_indices:
                # No overlapping boxes; add the box as is
                final_boxes.append(box)
                final_scores.append(score)
                final_classes.append(cls)
                used_indices.add(i)
            else:
                # Aggregate overlapping boxes
                overlapping_boxes = [all_boxes[j] * all_scores[j] for j in group_indices]
                overlapping_scores = [all_scores[j] for j in group_indices]

                # Weighted average for the final box
                avg_box = np.sum(overlapping_boxes, axis=0) / np.sum(overlapping_scores)

                # Average score, optionally scaled by model agreement
                avg_score = np.mean(overlapping_scores) * (len(group_indices) / 2)

                # Append aggregated results
                final_boxes.append(avg_box)
                final_scores.append(avg_score)
                # Select the class of the first box in the group for simplicity
                final_classes.append(cls)
                used_indices.update(group_indices)  # Mark all grouped indices as used

        output = {
            "boxes": np.array(final_boxes),
            "scores": np.array(final_scores),
            "classes": np.array(final_classes)
        }
        return output