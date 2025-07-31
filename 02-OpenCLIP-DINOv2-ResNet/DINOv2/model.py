from transformers import Dinov2ForImageClassification


def get_model(num_classes, device):
    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base", num_labels=num_classes, ignore_mismatched_sizes=True
    )
    model.to(device)
    return model
