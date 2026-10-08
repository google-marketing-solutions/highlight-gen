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

## Terms

HighlightGen is used with the following products and models, which are subject
to their own terms:

*   [Google Merchant Center Terms of Service](https://support.google.com/merchants/answer/160173)
*   [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

## License

Apache 2.0. See [LICENSE](LICENSE).
