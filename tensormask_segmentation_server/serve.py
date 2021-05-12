"""
Server for obtaining integrated and integrated directional gradients from specific image models.
"""

import click
from collections import OrderedDict
from flask import Flask, request, send_file
import gzip
from io import BytesIO
import numpy as np
import shutil
import torch


from . import cli_main
from .load_models import load_models
from .run_models import run_models

app = Flask(__name__)
MODEL_DICT = {}
DEVICE = None

@app.route("/model/", methods=["POST"])
def run_model():
    """
    Obtain the gradients from running the specified model on the input image tensor.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, integrated_directional_gradients, step_sizes, intermediates.
    """
    if request.method == 'POST':
        print(request)
        data = request.json

        img_data = data["img_data"]

        img = np.array(img_data, dtype=np.uint8)

        preds_dict = run_models(MODEL_DICT["model_name"],
                                MODEL_DICT["model"],
                                DEVICE,
                                img)

        temp_bytes, temp_gzip = BytesIO(), BytesIO()

        torch.save(preds_dict, temp_bytes)
        temp_bytes.seek(0)

        with gzip.GzipFile(fileobj=temp_gzip, mode='wb') as f_out:
            shutil.copyfileobj(temp_bytes, f_out)

        temp_gzip.seek(0)

        return send_file(temp_gzip, as_attachment=True, mimetype="/application/gzip", attachment_filename="returned_gradients.gzip")


@cli_main.command(help="Start a server and initialize the models for calculating image segmentation.")
@click.option(
    "-h",
    "--host",
    required=False,
    default="localhost",
    help="Host to bind to. Default localhost"
)
@click.option(
    "-p",
    "--port",
    default=8888,
    required=False,
    help="Port to bind to. Default 8888"
)
@click.option(
    "--tensormask-path",
    "-tp",
    required=True,
    help="Path to the pretrained TensorMask model",
    default=None,
)
@click.option(
    "--configuration-path",
    "-cp",
    required=True,
    help="Path to the model's configuration file.  Located in the tensormask_segmentation_server/configs folder.",
    default=None,
)
def serve(
        host,
        port,
        tensormask_path,
        configuration_path,
):
    global MODEL_DICT, DEVICE

    DEVICE = torch.device("cuda:0")
    # TensorMask will always load inputs on this device
    try:
        MODEL_DICT = load_models(DEVICE, tensormask_path, configuration_path)
    except Exception as e:
        print("An Error occurred: ", e)
        raise e

    app.run(host=host, port=port, debug=True)
