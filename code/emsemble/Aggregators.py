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
        # Unpack predictions, including model names
        boxes_list, scores_list, classes_list, model_names = zip(*predictions)

        all_boxes = np.vstack(boxes_list)
        all_scores = np.concatenate(scores_list)
        all_classes = np.concatenate(classes_list)
        all_model_names = sum([[name] * len(cls_list) for name, cls_list in zip(model_names, classes_list)], [])

        all_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        iou_matrix = box_iou(all_boxes_tensor, all_boxes_tensor)

        final_boxes, final_scores, final_classes = [], [], []
        used_indices = set()

        for i, (box, score, cls) in enumerate(zip(all_boxes, all_scores, all_classes)):
            if i in used_indices:
                continue

            overlaps = (iou_matrix[i] > iou_threshold).nonzero(as_tuple=False).flatten()
            group_indices = [j.item() for j in overlaps if j.item() not in used_indices]

            if not group_indices:
                final_boxes.append(box)
                final_scores.append(score)
                final_classes.append((cls, all_model_names[i]))  # Store as (class_id, model_name)
                used_indices.add(i)
            else:
                overlapping_boxes = [all_boxes[j] * all_scores[j] for j in group_indices]
                overlapping_scores = [all_scores[j] for j in group_indices]
                overlapping_models = [all_model_names[j] for j in group_indices]

                avg_box = np.sum(overlapping_boxes, axis=0) / np.sum(overlapping_scores)
                avg_score = np.mean(overlapping_scores) * (len(group_indices) / 2)

                final_boxes.append(avg_box)
                final_scores.append(avg_score)
                final_classes.append((cls, overlapping_models[0]))  # Use the first model in group
                used_indices.update(group_indices)

        output = {
            "boxes": np.array(final_boxes),
            "scores": np.array(final_scores),
            "classes": final_classes  # Now an array of (class_id, model_name)
        }
        return output