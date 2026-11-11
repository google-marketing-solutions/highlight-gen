# HighlightGen

**This is not an officially supported Google product.**

Eligibility for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security) is determined by
the [Google Open Source Software Vulnerability Reward Program
Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

This project is intended for demonstration purposes only. It is not intended
for use in a production environment.

## Problem

Comparison Shopping Service (CSS) inventory shows low adoption of the
`product_highlight` attribute. A proof of concept is required to show CSSs how
to generate product highlights based on a product's image, title and
description.

## Solution

HighlightGen is a standalone command-line tool that generates product
highlights from a product's image, title and description. It runs on your own
infrastructure, using a locally hosted open-weights
[Gemma](https://ai.google.dev/gemma) model, against your own Merchant Center
product data. No product data is sent to Google.

## Getting started

### Requirements

*   Python 3.11 or later (tested on 3.13).
*   A [Kaggle](https://www.kaggle.com/) account that has accepted the
    [Gemma terms](https://www.kaggle.com/models/google/gemma-4), with
    [API credentials](https://github.com/Kaggle/kagglehub#authenticate)
    available to `kagglehub`. The model is downloaded on first run. Pass
    `--model_path` instead to use weights you have already downloaded.
*   Any platform PyTorch supports. A GPU makes generation much faster, but is
    not required.

### Install

```sh
python -m venv .venv
source .venv/bin/activate
pip install .
```

`pip install .` installs the dependencies listed in `pyproject.toml` and adds a
`highlight-gen` command to the virtualenv. By default pip installs the standard
`torch` build for your operating system. To use a specific accelerator (CUDA,
ROCm) or a smaller CPU-only build, install `torch` first by following the
[PyTorch instructions](https://pytorch.org/get-started/locally/) for your
platform, then run `pip install .`; pip keeps the `torch` you chose.

HighlightGen does not ship a lockfile, because `torch` builds differ by platform
and a pinned, hash-verified requirements file would only install on one of them.
To produce one for your own platform, run
`pip-compile --generate-hashes -o requirements.txt pyproject.toml` from
[pip-tools](https://pip-tools.readthedocs.io/).

### Run

```sh
python highlight_gen.py --input_file=feed.csv --output_file=feed_out.tsv
```

The input is a Merchant Center product feed in CSV or TSV format. The output is
the same feed, written as TSV, with a `product_highlight` column populated. Run
`python highlight_gen.py --help` for all flags, including `--resume` to continue
an interrupted run and `--num_rows` to try a sample first.

On first use, transformers 5.18 logs a one-time deprecation warning about
`generation_config`. It comes from inside the transformers pipeline and is
harmless.

### Test

```sh
python highlight_gen_test.py
```

## Terms

HighlightGen is used with the following products and models, which are subject
to their own terms:

*   [Google Merchant Center Terms of Service](https://support.google.com/merchants/answer/160173)
*   [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

## Authors

*   Max Reemets ([@Max-Reemets](https://github.com/Max-Reemets)), who wrote the
    original script.
*   John Stafford ([@jwstafford](https://github.com/jwstafford))

## License

Apache 2.0. See [LICENSE](LICENSE).
