This is a fork of [Facebook's Side project](https://github.com/facebookresearch/side) to showcase the verification engine and enable live checking of citations on Wikipedia without relying on the pre-processed WAFER data.

## Code
* `verify_wikipedia`: original Side code (`conf/` and `/dpr`) but streamlined to only include code relevant to running the verification engine (verifier model) -- i.e. I removed code for building the Sphere indexes and training models.
I also added code for running a Flask API for running the verifier model (`wsgi.py`), fetch passages from external sources (`passages/web_source.py`), and fetch claims from Wikipedia articles (`passages/wiki_claim.py`).
* `api_config`: Scripts / config for setting up the verifier model Flask API on a Cloud VPS server.
* `tool_ui`: code for Flask-based UI for calling the verififer model to be hosted on Toolforge.

## External data dependencies
* The model itself can be downloaded per
[the original README](https://github.com/facebookresearch/side/tree/main/projects/verify_wikipedia#downloading-index-and-models)
from [this URL (8.0GB)](https://dl.fbaipublicfiles.com/side/verifier.tar.gz). I have not modified the model. That file contains a few important pieces:
  * `verifier/outputs/checkpoint.best_validation_acc`: model weights (notably, you can delete `checkpoint.best_validation_loss` and free up 4GB)
  * `verifier/predictions/best_validation_acc__wafer_ccnet/checkpoint_cfg.yaml`: model config
  * `verifier/.hydra/`: more config
* Running the verifier model for the first time will automatically download the base [HuggingFace Roberta-large model (1.3GB)](https://huggingface.co/roberta-large/blob/main/pytorch_model.bin) and cache it locally.

## Test it out
* Tool UI can be found at: https://citation-evaluation.toolforge.org/