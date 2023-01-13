# Flask API that exposes the Verification Engine (verifier) from Facebook's Side project.
#
# Background: https://github.com/facebookresearch/side
# Goal: Help prioritize citations on English Wikipedia for verification / improvement
#
# Components:
# * web_source: gather passages from a given external URL for verification
# * wiki_claim: extract claims (text + citation URL supposedly supporting it) from a Wikipedia article
# * verifier: Side model for comparing a passage from a web source with a Wikipedia claim and computing the support
#    * Higher scores indicate more support for the claim from the given passage
#    * There is no threshold at which one can say the passage does or does not support the claim -- best used for ranking
#    * Verifier takes ~20 seconds to load initially and 1.5 seconds to verify a single claim on CPUs
#
# API Endpoints:
# * /api/verify-random-claim: explore the model -- fetch a random citation from a Wikipedia article and evaluate it
# * /api/get-all-claims: generate input data -- get all claims for a Wikipedia article
# * /api/verify-claim: verify a single claim -- check a claim from get-all-claims

import logging
import os
import random
import sys
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml

import hydra
from hydra import compose, initialize
import mwapi
from omegaconf import OmegaConf
import torch

__dir__ = os.path.dirname(__file__)
__updir = os.path.abspath(os.path.join(__dir__, '..'))
sys.path.append(__updir)
sys.path.append(__dir__)

from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj
from passages.web_source import get_passages
from passages.wiki_claim import get_claims

app = Flask(__name__)

# load in app user-agent or any other app config
app.config.update(
    yaml.safe_load(open(os.path.join(__updir, 'flask_config.yaml'))))

# Enable CORS for API endpoints
cors = CORS(app, resources={r'/api/*': {'origins': '*'}})

# Global variables to hold model and other supports
CTX_SRC = None
BIENCODER_PREPARE_ENCODER_INPUTS_FUNC = None
BIENCODER = None

@app.route('/api/verify-random-claim', methods=['GET'])
def verify_random_claim():
    page_title, error = validate_api_args()
    if error is not None:
        return jsonify({'error': error})
    else:
        claims = get_claims(title=page_title, user_agent=app.config['CUSTOM_UA'])
        if claims:
            claim = random.choice(claims)
            url, section, text = claim
            result = {'article': f'https://en.wikipedia.org/wiki/{page_title}',
                      'claim': {'url':url, 'section':section, 'text':text},
                      'passages':[]
                      }
            for passage in get_passages(url=url, user_agent=app.config['CUSTOM_UA']):
                if passage is not None:
                    start = time.time()
                    source_title, passage_text = passage
                    score = get_score(text, {"title": source_title, "text": passage_text, "url": url})
                    result['source_title'] = source_title
                    result['passages'].append({'passage':passage_text, 'score':score, 'time (s)':time.time() - start})
            return jsonify(result)
        else:
            return jsonify({'error':f'no verifiable claims for https://en.wikipedia.org/wiki/{page_title}'})

@app.route('/api/get-all-claims', methods=['GET'])
def get_all_claims():
    page_title, error = validate_api_args()
    if error is not None:
        return jsonify({'error': error})
    else:
        claims = get_claims(title=page_title, user_agent=app.config['CUSTOM_UA'])
        result = {'article': f'https://en.wikipedia.org/wiki/{page_title}',
                  'claims': [{'url': c[0], 'section': c[1], 'text': c[2]} for c in claims]
                  }
        return jsonify(result)


@app.route('/api/verify-claim', methods=['POST'])
def verify_claim():
    """Verify a claim.

    Fields:
    * wiki_claim (str): passage from a Wikipedia article that crucially contains a [CIT] token indicating where a citation occurs to be evaluated
        * Only text before first [CIT] token is considered; if no [CIT] token then full passage considered.
        * Claim is expected in the form of "<article title> [SEP] <section title> [SEP] <pre-citation passage> [CIT] <post-citation passage>"
        * Pre-citation passages should generally be ~150 words.
    * source_url (str): source URL from which passages are fetched to score.
    """
    try:
        wiki_claim = request.form['wiki_claim']
        source_url = request.form['source_url']
    except KeyError:
        return jsonify({'error':f'Received {request.form.keys()} but expected the following fields: wiki_claim: str, source_url: str'})

    result = {'passages':[]}
    pass_idx = 0
    for passage in get_passages(url=source_url, user_agent=app.config['CUSTOM_UA']):
        if passage is not None:
            pass_idx += 1
            start = time.time()
            source_title, passage_text = passage
            score = get_score(wiki_claim, {"title": source_title, "text": passage_text, "url": source_url})
            result['source_title'] = source_title
            result['passages'].append({'passage': passage_text, 'score': score,
                                       'idx':pass_idx, 'time (s)': time.time() - start})

    # Rank from most to least support
    result['passages'] = sorted(result['passages'], key=lambda x: x.get('score', -1), reverse=True)
    return jsonify(result)

def get_crossencoder_components_from_checkpoint(cfg):
    """Load in verification model from disk.

    Taken from:
    * https://github.com/facebookresearch/side/blob/verify_wikipedia/projects/verify_wikipedia/misc/inference_notebook.py
    """
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
    """Score the support of a claim from a given passage.

    Taken from:
    * https://github.com/facebookresearch/side/blob/verify_wikipedia/projects/verify_wikipedia/notebooks/Reranker%20inference.ipynb
    """
    question_ids = CTX_SRC.preprocess_query(wiki_claim, training=False)
    ctx_ids = CTX_SRC.preprocess_passage(CTX_SRC.passage_struct_from_dict(passage), training=False)
    inputs = {
        "question_ids": question_ids.view(1, -1),
        "question_segments": torch.zeros_like(question_ids),
        "context_ids": ctx_ids.view(1, -1),
        "ctx_segments": torch.zeros_like(ctx_ids),
    }
    inputs = BIENCODER_PREPARE_ENCODER_INPUTS_FUNC(inputs, CTX_SRC)
    with torch.no_grad():
        _, score = BIENCODER(
            **inputs,
            encoder_type="question",
            representation_token_pos=0,
        )
    return score.item()

def get_canonical_page_title(title, session=None):
    """Resolve redirects / normalization -- used to verify that an input page_title exists"""
    if session is None:
        session = mwapi.Session('https://en.wikipedia.org', user_agent=app.config['CUSTOM_UA'])

    result = session.get(
        action="query",
        prop="info",
        inprop='',
        redirects='',
        titles=title,
        format='json',
        formatversion=2
    )
    if 'missing' in result['query']['pages'][0]:
        return None
    else:
        return result['query']['pages'][0]['title']

def validate_api_args():
    """Validate API arguments for language-agnostic model."""
    error = None
    page_title = None
    if request.args.get('title'):
        page_title = get_canonical_page_title(request.args['title'])
        if page_title is None:
            error = f'no matching article for "https://en.wikipedia.org/wiki/{request.args["title"]}"'
    else:
        error = 'missing title -- e.g., "2005_World_Series" for "https://en.wikipedia.org/wiki/2005_World_Series"'

    return page_title, error


def load_model():
    """Start-up to load in model and make sure working alright."""
    global CTX_SRC, BIENCODER_PREPARE_ENCODER_INPUTS_FUNC, BIENCODER
    start = time.time()
    CTX_SRC, BIENCODER_PREPARE_ENCODER_INPUTS_FUNC, BIENCODER = \
        get_crossencoder_components_from_checkpoint([
            "retriever=reranker",
            "retriever.ctx_src=wafer_ccnet",
            "retriever.batch_size=2",
            "retriever.checkpoint_dir='/extrastorage/verifier'",
        ])

    test_model()
    logging.info(f'{time.time() - start:.1f} seconds for model loading.')

def test_model():
    """Make sure model loads properly and outputs as expected."""
    from dpr.options import logger, setup_logger
    setup_logger(logger)

    # Example Wikipedia claim
    ex_input = "University of Essex [SEP] Section::::Reputation.:Rankings. [SEP] Research Excellence Framework (REF 2014) and has been in the top 15 for overall student satisfaction six years running, amongst mainstream English universities, according to the National Student Survey (NSS, 2018).\n The 1987 Nobel Peace Prize was awarded to \u00d3scar Arias who completed his doctorate in Political Science in 1973. The 2010 Nobel Prize for Economics was awarded to Christopher Pissarides who gained his BA and MA degrees in Economics in the early 1970s. In 2016 former Essex academic Oliver Hart won the Nobel Prize for Economics. Derek Walcott, who received the 1992 Nobel Prize in Literature, served as Professor of Poetry at the university from 2010 to 2013 before his retirement.\n Section::::Reputation.:Rankings.\nNationally, Essex was ranked 29th overall in \"The Times and The Sunday Times Good University Guide\" 2019[CIT] and was shortlisted for its University of the Year Award in 2018.\nEssex was rated Gold in the Teaching Excellence Framework in 2017. The TEF Panel noted students from all backgrounds achieved outstanding outcomes with regards to continuation and progression to highly skilled employment or further study and outstanding levels of satisfaction with teaching, academic support, and assessment and feedback.\n Essex has been consistently ranked first for politics research and was once again ranked top in the Research Excellence Framework 2014 (REF2014) for politics and international studies. Essex was 19th overall, out of mainstream UK universities, according to the Times Higher Education's 'intensity' ranking for REF2014 which mapped university performance against the proportion of eligible staff submitted. Nine Essex subjects were ranked in the top"
    # Example external source title
    ex_title = "Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide | University of Essex"
    # Example external source passage
    ex_passage = "Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide | University of Essex News Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide Essex ranked in Top 30 in The Times and The Sunday Times Good University Guide The University of Essex has been ranked 29th in The Times and The Sunday Times Good University Guide 2019. The Good University Guide provides the definitive rankings for UK universities and the most comprehensive overview of higher education in Britain. The Guide noted that Essex had seen applications rise in 2018 for the fourth year in a row, at a time where other universities are seeing demand fall. Essex"
    # Example external source URL
    ex_url = "https://www.essex.ac.uk/news/2018/09/21/essex-ranked-in-top-30-in-the-times-and-the-sunday-times-good-university-guide"
    # Expected score (based on original WAFER results)
    ex_expected_score = 27.0658

    score = get_score(ex_input, {"title": ex_title, "text": ex_passage, "url": ex_url})
    assert abs(score - ex_expected_score) < 0.001, score
    logging.info("Model working as expected.")

load_model()
application = app

if __name__ == '__main__':
    app.run()