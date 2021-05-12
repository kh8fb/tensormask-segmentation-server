"""
Script for converting an input image into JSON.
"""

from PIL import Image
import json
import sys

def convert_img_to_json(img_path, classification, save_path):
    """
    Converts an input image into JSON RGB representation.

    Parameters
    ----------
    img_path: str
        Path to the input image to prepare for modeling.
    classification: str
        String label of the image.
        One of ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’,
        ‘horse’, ‘ship’, ‘truck’
    save_path: str
        File location where final JSON output will be saved.

    Returns
    -------
    None
    """
    img = Image.open(img_path)
    
    json_dict = {
        "img_data": list(img.getdata()),
        "target_class": classification
    }
    
    with open(save_path, "w") as fobj:
        json.dump(json_dict, fobj)
    return


def main():
    convert_img_to_json(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))

if __name__ == "__main__":
    main()
    
