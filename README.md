This is a fork of Facebook's Side project: https://github.com/facebookresearch/side

I have isolated just the code necessary for comparing a passage with a citation and a passage from an external website
to determine how well the citation supports the text. This means I have removed a lot of code related to training and
other parameterizations of the models.

I have also added a script `verify.py` that processes a single example pair of citation passage and source passage
to show how the model might be used to check new citations.

External dependencies:
* The model itself can be downloaded per
[the original README](https://github.com/facebookresearch/side/tree/main/projects/verify_wikipedia#downloading-index-and-models)
from [this URL (8.0GB)](https://dl.fbaipublicfiles.com/side/verifier.tar.gz). That file contains a few important pieces:
  * `verifier/outputs/checkpoint.best_validation_acc`: model weights (notably, you can delete `checkpoint.best_validation_loss` and free up 4GB)
  * `verifier/predictions/best_validation_acc__wafer_ccnet/checkpoint_cfg.yaml`: model config
  * `verifier/.hydra/`: more config
* [HuggingFace Roberta-large model (1.3GB)](https://huggingface.co/roberta-large/blob/main/pytorch_model.bin)

External components to make this a complete system:
* Fetching HTML for a given URL and parsing into passages: https://public.paws.wmcloud.org/User:Isaac_(WMF)/side/html_to_passages.ipynb
* Fetching a citation from a Wikipedia article and parsing into a passage + URL: TODO
* API for combining the components together: TODO


TODOs:
* Improve README
* Add pre-commit
* Page Report:
  * API call for all citations (text, context like paragraph/template/list, url)
  * Present citations as table w/ best-passage + score columns too
  * For each citation that meets criteria, check with verifier and add score/best-passage
* Get Cloud VPS instance for API
* Get Toolforge instance for page report
* Can eventually connect that up with page-create stream / cache db