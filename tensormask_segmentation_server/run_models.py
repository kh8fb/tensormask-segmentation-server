"""
Run the model and get the pred_mask outputs.
"""

import torch


def prepare_input(img_array, device):
    """
    Crop and prepare the input for modeling.

    Parameters
    ----------
    img_array: np.array(3,x,x)
        RGB array of the photo to obtain classification from.
    device: torch.device
        Device that models are stored on.
    Returns
    -------
    input_dict: dict
        Dictionary with keys 'image', 'height', and 'width' that will be passed to the model.
    """
    input_tensor = torch.tensor(img_array) # torch.tensor(3, height, width)
    height = input_tensor.shape[1]
    width = input_tensor.shape[2]
    input_dict = {
        "image": input_tensor.to(device),
        "height": height,
        "width": width,
    }
    return input_dict

def run_models(model_name, model, device, img):
    """
    Run the model on the input image and return the tensor of prediction masks for image locations.

    Parameters
    ----------
    model_name: str
        Name of the model that is being run.
        Currently supported is "tensormask".
    model: torch.nn.Module
        Model to run .
    device: torch.device
        Device that models are stored on.
    img: np.array(3,x,y)
        BGR array of the photo to obtain segmentation from.
    Returns
    -------
    preds_dict: dict
        Dictionary containing the prediction masking tensors with the following keys:
            "pred_masks": torch.tensor(num_segmentations,height,width)
    """
    input_dict = prepare_input(img, device)
    if model_name == "tensormask":
        with torch.no_grad():
            outputs = model([input_dict])
        pred_masks = outputs[0]['instances'].pred_masks.cpu()
        preds_dict = {"pred_masks": pred_masks}

    return preds_dict
