# Sources:
# * https://github.com/facebookresearch/side/blob/verify_wikipedia/projects/verify_wikipedia/misc/inference_notebook.py
# * https://github.com/facebookresearch/side/blob/verify_wikipedia/projects/verify_wikipedia/notebooks/Reranker%20inference.ipynb
#
# Goal:
# * Inputs:
#   * Claim: Wikipedia sentences representing a fact
#   * Model: fine-tuned Robert-Large model
#   * Source: Passage from cited source
# * Output: score for that passage

import hydra
from hydra import compose, initialize

from omegaconf import OmegaConf
import torch

from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj

def get_crossencoder_components_from_checkpoint(cfg):

    initialize(config_path="./conf/")
    cfg = compose(
        config_name="retrieval",
        overrides=cfg,
    )

    # print(OmegaConf.to_yaml(cfg))

    # Load all the configs
    checkpoint_config = OmegaConf.load(f"{cfg.retriever.checkpoint_dir}/.hydra/config.yaml")
    checkpoint_cfg = setup_cfg_gpu(checkpoint_config)

    # Load weights
    saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}")
    set_cfg_params_from_state(saved_state.encoder_params, checkpoint_cfg)

    # Get model and tensorizer
    tensorizer, biencoder, _ = init_biencoder_components(
        checkpoint_cfg.base_model.encoder_model_type, checkpoint_cfg, inference_only=True
    )

    # Set model to eval
    biencoder.eval()

    # load weights from the model file
    biencoder = get_model_obj(biencoder)
    biencoder.load_state(saved_state, strict=True)

    # Instantiate dataset
    ctx_src = hydra.utils.instantiate(cfg.retriever.datasets[cfg.retriever.ctx_src], checkpoint_cfg, tensorizer)

    # Get helper functions
    loss_func = biencoder.get_loss_function()
    biencoder_prepare_encoder_inputs_func = biencoder.prepare_model_inputs

    return ctx_src, biencoder_prepare_encoder_inputs_func, biencoder

def get_score(wiki_claim, passage):
    ctx_src, biencoder_prepare_encoder_inputs_func, biencoder = \
        get_crossencoder_components_from_checkpoint([
            "retriever=reranker",
            "retriever.ctx_src=wafer_ccnet",
            "retriever.batch_size=2",
            "retriever.checkpoint_dir='/Users/ijohnson/Downloads/verifier'",
        ])

    question_ids = ctx_src.preprocess_query(wiki_claim, training=False)
    ctx_ids = ctx_src.preprocess_passage(ctx_src.passage_struct_from_dict(passage), training=False)
    inputs = {
        "question_ids": question_ids.view(1, -1),
        "question_segments": torch.zeros_like(question_ids),
        "context_ids": ctx_ids.view(1, -1),
        "ctx_segments": torch.zeros_like(ctx_ids),
    }
    inputs = biencoder_prepare_encoder_inputs_func(inputs, ctx_src)
    with torch.no_grad():
        _, score = biencoder(
            **inputs,
            encoder_type="question",
            representation_token_pos=0,
        )
    return score

def main():
    ex_input = "University of Essex [SEP] Section::::Reputation.:Rankings. [SEP] Research Excellence Framework (REF 2014) and has been in the top 15 for overall student satisfaction six years running, amongst mainstream English universities, according to the National Student Survey (NSS, 2018).\n The 1987 Nobel Peace Prize was awarded to \u00d3scar Arias who completed his doctorate in Political Science in 1973. The 2010 Nobel Prize for Economics was awarded to Christopher Pissarides who gained his BA and MA degrees in Economics in the early 1970s. In 2016 former Essex academic Oliver Hart won the Nobel Prize for Economics. Derek Walcott, who received the 1992 Nobel Prize in Literature, served as Professor of Poetry at the university from 2010 to 2013 before his retirement.\n Section::::Reputation.:Rankings.\nNationally, Essex was ranked 29th overall in \"The Times and The Sunday Times Good University Guide\" 2019[CIT] and was shortlisted for its University of the Year Award in 2018.\nEssex was rated Gold in the Teaching Excellence Framework in 2017. The TEF Panel noted students from all backgrounds achieved outstanding outcomes with regards to continuation and progression to highly skilled employment or further study and outstanding levels of satisfaction with teaching, academic support, and assessment and feedback.\n Essex has been consistently ranked first for politics research and was once again ranked top in the Research Excellence Framework 2014 (REF2014) for politics and international studies. Essex was 19th overall, out of mainstream UK universities, according to the Times Higher Education's 'intensity' ranking for REF2014 which mapped university performance against the proportion of eligible staff submitted. Nine Essex subjects were ranked in the top"
    ex_title = "Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide | University of Essex"
    ex_passage = "Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide | University of Essex News Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide The University of Essex has been ranked 29th in The Times and The Sunday Times Good University Guide 2019. The Good University Guide provides the definitive rankings for UK universities and the most comprehensive overview of higher education in Britain. The Guide noted that Essex had seen applications rise in 2018 for the fourth year in a row, at a time where other universities are seeing demand fall. Essex"
    ex_url = "n/a"
    print(get_score(
        ex_input,
        {
            "title": ex_title,
            "text": ex_passage,
            "url": ex_url,
        },
    ))

if __name__ == "__main__":
    main()