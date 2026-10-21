# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates Merchant Center product highlights from product images and text.

HighlightGen reads a product feed (CSV or TSV), sends each product's image
together with its title, description, colour, material and size to a locally
hosted Gemma model, and writes the feed back out as TSV with a
`product_highlight` column populated.

Inference runs entirely on the local machine. No product data is sent to
Google.

Example:
  python highlight_gen.py --input_file=feed.csv --output_file=feed_out.tsv
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import csv
import dataclasses
import io
import itertools
import os
import re
from typing import Any, TextIO, TypeAlias

from absl import app
from absl import flags
from absl import logging
import kagglehub
from PIL import Image
import requests
import tqdm
import transformers

__version__ = '0.1.0'

_INPUT_FILE = flags.DEFINE_string(
    'input_file',
    '',
    'Path to the input product feed (CSV or TSV).',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '',
    'Path to write the enriched feed to (always written as TSV).',
)
_NUM_ROWS = flags.DEFINE_integer(
    'num_rows',
    0,
    'Number of feed rows to process. 0 processes all rows.',
    lower_bound=0,
)
_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '',
    'Local directory containing pre-downloaded Gemma weights. If empty, the '
    'model is downloaded from Kaggle on first run.',
)

KAGGLE_MODEL_HANDLE = 'google/gemma-4/transformers/gemma-4-e2b-it'
HIGHLIGHT_COLUMN = 'product_highlight'

# Identifies the tool honestly to merchant CDNs. Requests that are refused
# (e.g. HTTP 403) are skipped and logged rather than retried with a
# different identity.
USER_AGENT = (
    f'HighlightGen/{__version__} '
    '(+https://github.com/google-marketing-solutions/highlight-gen)'
)
IMAGE_REQUEST_TIMEOUT_S = 10
# Product images are rarely more than a few MB. The cap stops a single
# oversized or misconfigured URL from exhausting memory.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_DOWNLOAD_CHUNK_BYTES = 64 * 1024
MAX_NEW_TOKENS = 512

# Merchant Center limits, see
# https://support.google.com/merchants/answer/9216100.
MIN_HIGHLIGHTS = 2
MAX_HIGHLIGHTS = 5
MAX_HIGHLIGHT_LENGTH = 150

_PRODUCT_DETAIL_COLUMNS = ('title', 'description', 'color', 'material', 'size')

# Separates several URLs in one image field: whitespace, or a comma directly
# before another URL. Other commas are kept, because image CDNs use them inside
# URLs (e.g. Cloudinary's .../upload/w_800,h_800/shoe.jpg).
_URL_SEPARATOR = re.compile(r'\s*,\s*(?=[a-zA-Z][a-zA-Z0-9+.-]*://)|\s+')

# Candidate feed delimiters, detected from the header line.
_FEED_DELIMITERS = ',\t;|'

# Tabs and line breaks inside values would break the TSV structure.
_LINE_BREAKS_AND_TABS = re.compile(r'[\t\r\n]+')

# Model control tokens are not listed here: the transformers pipeline removes
# them when decoding (`skip_special_tokens` defaults to True). Markdown bold is
# ordinary text, so it is removed separately in `clean_model_output`.
_MARKDOWN_BOLD = '**'
_PREAMBLE_PATTERN = re.compile(r'^Here are .*?:', flags=re.IGNORECASE)
# Bullets "*" and "•" never start a real highlight, so they are removed with or
# without a following space. "-" and numbering such as "1." or "2)" only count
# as list markers when followed by a space, so values such as "1.5mm steel" and
# "-20°C rated" are kept intact.
_LIST_MARKER_PATTERN = re.compile(r'^\s*(?:[*•]\s*|-\s+|\d+[.)]\s+)')
# Highlights in the model output are separated by commas or line breaks.
_HIGHLIGHT_SEPARATOR = re.compile(r'[,\n]')

_PROMPT_INTRO = (
    'You are an expert merchandiser and copywriter. Analyze the product image'
    ' and the following details:'
)
_PROMPT_INSTRUCTIONS = (
    'Generate 3 to 5 concise, comma-separated highlights that showcase the'
    " product's most compelling features and benefits for a potential"
    ' customer. Focus on what makes this product stand out based on the image'
    ' and text provided. DO NOT use bullets or numbered list. Make sure there'
    ' are no extra commas. Commas must only separate the highlights.'
)

# Callable that sends the chat messages for one product to the model and
# returns the transformers pipeline output.
TextGenerator: TypeAlias = Callable[[list[dict[str, Any]]], Any]
# Callable with the signature of `requests.get`.
HttpGet: TypeAlias = Callable[..., requests.Response]


class ImageFetchError(Exception):
  """Raised when a product image cannot be downloaded or decoded."""


@dataclasses.dataclass(frozen=True)
class HighlightResult:
  """The outcome of generating highlights for one product.

  Attributes:
    highlights: The validated highlights. Empty if generation failed.
    error: Why generation failed, or an empty string on success.
  """

  highlights: Sequence[str]
  error: str = ''


@dataclasses.dataclass(frozen=True)
class FeedSummary:
  """Counts from processing a feed.

  Attributes:
    processed_count: Number of products processed.
    failed_count: Number of processed products left without highlights.
    skipped_line_count: Number of malformed input lines that were skipped.
  """

  processed_count: int
  failed_count: int
  skipped_line_count: int


def _detect_delimiter(header_line: str, source_name: str) -> str:
  """Returns the most frequent candidate delimiter in a feed's header line.

  The header is used because column names rarely contain delimiters, whereas
  data rows can be malformed, which defeats `csv.Sniffer`.

  Args:
    header_line: The first line of the feed.
    source_name: Name of the feed, for the error message.

  Returns:
    The detected delimiter.

  Raises:
    ValueError: If the header contains none of the candidate delimiters.
  """
  delimiter = max(_FEED_DELIMITERS, key=header_line.count)
  if not header_line.count(delimiter):
    raise ValueError(
        f'Could not detect the delimiter of {source_name}: the header line'
        f' contains none of {list(_FEED_DELIMITERS)}. Expected a CSV or TSV'
        ' file.'
    )
  return delimiter


class FeedReader:
  """Streams products from a CSV or TSV feed, one row at a time.

  The delimiter is detected from the header line. Lines with more fields than
  the header are skipped and logged, since there is no safe way to tell which
  value belongs to which column. Lines with fewer fields are kept, and the
  missing trailing values are treated as empty. Blank lines are ignored.

  Attributes:
    columns: Column names from the header row, stripped and lower-cased.
    skipped_line_count: Number of malformed lines skipped so far.
  """

  def __init__(self, feed_file: TextIO, source_name: str):
    """Initializes the reader and reads the header row.

    Args:
      feed_file: The open feed file. Must be seekable.
      source_name: Name of the feed, used in log and error messages.

    Raises:
      ValueError: If the feed's delimiter cannot be detected.
    """
    self._source_name = source_name
    delimiter = _detect_delimiter(feed_file.readline(), source_name)
    feed_file.seek(0)
    self._reader = csv.reader(feed_file, delimiter=delimiter)
    self.columns = [column.strip().lower() for column in next(self._reader)]
    self.skipped_line_count = 0

  def __iter__(self) -> Iterator[dict[str, str]]:
    """Yields each product as a mapping from column name to value.

    Yields:
      One dict per product, keyed by the lower-cased column names. Missing
      trailing values are empty strings.
    """
    for fields in self._reader:
      if not any(field.strip() for field in fields):
        continue
      if len(fields) > len(self.columns):
        self.skipped_line_count += 1
        logging.warning(
            'Skipped line %d of %s: expected %d fields, found %d.',
            self._reader.line_num,
            self._source_name,
            len(self.columns),
            len(fields),
        )
        continue
      fields += [''] * (len(self.columns) - len(fields))
      yield dict(zip(self.columns, fields))


class TsvWriter:
  """Writes products to a Merchant Center tab-separated file, one at a time.

  Each row is flushed as soon as it is written, so the output keeps every
  finished product if a long run stops part-way through.

  Fields are written without quoting so that the quoting inside
  `product_highlight` values reaches Merchant Center unchanged. Tabs and line
  breaks inside values are replaced with spaces, since they would otherwise
  break the file structure.
  """

  def __init__(self, output_file: TextIO, columns: Sequence[str]):
    """Initializes the writer and writes the header row.

    Args:
      output_file: The open output file.
      columns: Column names, in output order.
    """
    self._output_file = output_file
    self._columns = columns
    self._writer = csv.writer(
        output_file,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
        quotechar=None,
        escapechar=None,
        lineterminator='\n',
    )
    self._write_fields(columns)

  def write_row(self, row: Mapping[str, str]) -> None:
    """Writes one product, using empty values for any missing columns."""
    self._write_fields([row.get(column, '') for column in self._columns])

  def _write_fields(self, fields: Sequence[str]) -> None:
    self._writer.writerow(
        _LINE_BREAKS_AND_TABS.sub(' ', field) for field in fields
    )
    self._output_file.flush()


def select_image_url(product: Mapping[str, str]) -> str:
  """Returns the first image URL for a product.

  Uses the first HTTP(S) URL in `image_link`, falling back to
  `additional_image_link` when `image_link` has none. Either field may
  contain several URLs, separated by whitespace or by a comma directly before
  the next URL.

  Args:
    product: One feed row.

  Returns:
    The first HTTP(S) URL found, or an empty string if there is none.
  """
  for column in ('image_link', 'additional_image_link'):
    for candidate in _URL_SEPARATOR.split(product.get(column, '').strip()):
      if candidate.lower().startswith(('http://', 'https://')):
        return candidate
  return ''


def _image_too_large_error(url: str) -> ImageFetchError:
  """Returns the error for an image larger than `MAX_IMAGE_BYTES`."""
  return ImageFetchError(
      f'Image {url} is larger than the'
      f' {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit'
  )


def _read_image_bytes(response: requests.Response, url: str) -> bytes:
  """Reads a streamed response body, stopping once it exceeds the size cap.

  Args:
    response: A response opened with `stream=True`.
    url: The image URL, for the error message.

  Returns:
    The response body.

  Raises:
    ImageFetchError: If the body is larger than `MAX_IMAGE_BYTES`.
  """
  # Reject early when the server declares the size up front.
  declared_size = response.headers.get('Content-Length', '')
  if declared_size.isdigit() and int(declared_size) > MAX_IMAGE_BYTES:
    raise _image_too_large_error(url)
  # The declared size can be missing or wrong, so also count while reading.
  body = bytearray()
  for chunk in response.iter_content(chunk_size=_IMAGE_DOWNLOAD_CHUNK_BYTES):
    body += chunk
    if len(body) > MAX_IMAGE_BYTES:
      raise _image_too_large_error(url)
  return bytes(body)


def fetch_image(url: str, http_get: HttpGet = requests.get) -> Image.Image:
  """Downloads and decodes a product image.

  Args:
    url: The image URL.
    http_get: Function used to make the request. Defaults to `requests.get`.

  Returns:
    The decoded image in RGB mode.

  Raises:
    ImageFetchError: If the request fails or is refused, the image is larger
      than `MAX_IMAGE_BYTES`, or the data is not a decodable image.
  """
  try:
    response = http_get(
        url,
        headers={'User-Agent': USER_AGENT},
        timeout=IMAGE_REQUEST_TIMEOUT_S,
        stream=True,
    )
    # Closing the response releases the connection even if reading stops
    # early.
    with response:
      response.raise_for_status()
      content = _read_image_bytes(response, url)
  except requests.RequestException as e:
    raise ImageFetchError(f'Could not download image {url}: {e}') from e
  try:
    return Image.open(io.BytesIO(content)).convert('RGB')
  except (
      Image.UnidentifiedImageError,
      Image.DecompressionBombError,
      OSError,
      ValueError,
  ) as e:
    raise ImageFetchError(f'Could not decode image {url}: {e}') from e


def build_messages(
    product: Mapping[str, str], image: Image.Image
) -> list[dict[str, Any]]:
  """Builds the multimodal chat prompt for one product.

  Args:
    product: One feed row.
    image: The product image.

  Returns:
    Chat messages in the format expected by the transformers
    `image-text-to-text` pipeline.
  """
  details = []
  for column in _PRODUCT_DETAIL_COLUMNS:
    value = product.get(column, '').strip()
    if value:
      details.append(f'{column.capitalize()}: {value}')
  return [{
      'role': 'user',
      'content': [
          {'type': 'image', 'image': image},
          {'type': 'text', 'text': _PROMPT_INTRO},
          {'type': 'text', 'text': '\n'.join(details) + '\n'},
          {'type': 'text', 'text': _PROMPT_INSTRUCTIONS},
      ],
  }]


def clean_model_output(text: str) -> str:
  """Removes markdown bold and any preamble from the model output.

  Args:
    text: Generated text, already decoded without control tokens.

  Returns:
    The cleaned text.
  """
  text = text.replace(_MARKDOWN_BOLD, '')
  return _PREAMBLE_PATTERN.sub('', text.strip()).strip()


def parse_highlights(text: str) -> list[str]:
  """Splits cleaned model output into individual highlights.

  Drops empty values, list markers, duplicates and values longer than the
  Merchant Center limit, and keeps at most `MAX_HIGHLIGHTS` values.

  Args:
    text: Cleaned model output with comma-separated highlights.

  Returns:
    The highlights, in the order the model produced them.
  """
  highlights = []
  seen = set()
  for raw_value in _HIGHLIGHT_SEPARATOR.split(text):
    value = _LIST_MARKER_PATTERN.sub('', raw_value).strip().rstrip('.')
    if not value or len(value) > MAX_HIGHLIGHT_LENGTH:
      continue
    if value.lower() in seen:
      continue
    seen.add(value.lower())
    highlights.append(value)
  return highlights[:MAX_HIGHLIGHTS]


def format_highlights(highlights: Sequence[str]) -> str:
  """Formats highlights as a Merchant Center `product_highlight` value.

  Values are comma-separated. A value containing a comma, colon or double
  quote is wrapped in double quotes, with any double quotes inside it doubled,
  as Merchant Center requires for repeated attributes in text feeds.

  Args:
    highlights: The highlights to format.

  Returns:
    The formatted attribute value.
  """
  formatted = []
  for highlight in highlights:
    if any(char in highlight for char in ',:"'):
      escaped_highlight = highlight.replace('"', '""')
      formatted.append(f'"{escaped_highlight}"')
    else:
      formatted.append(highlight)
  return ','.join(formatted)


def generate_highlights(
    product: Mapping[str, str],
    generator: TextGenerator,
    http_get: HttpGet = requests.get,
) -> HighlightResult:
  """Generates highlights for one product.

  Args:
    product: One feed row.
    generator: Sends chat messages to the model.
    http_get: Function used to download the product image.

  Returns:
    The validated highlights, or the reason generation failed.
  """
  image_url = select_image_url(product)
  if not image_url:
    return HighlightResult(highlights=(), error='No image link provided')
  try:
    image = fetch_image(image_url, http_get)
  except ImageFetchError as e:
    return HighlightResult(highlights=(), error=str(e))

  try:
    output = generator(build_messages(product, image))
  except (RuntimeError, ValueError) as e:
    return HighlightResult(highlights=(), error=f'Model generation failed: {e}')

  highlights = parse_highlights(clean_model_output(output[0]['generated_text']))
  if len(highlights) < MIN_HIGHLIGHTS:
    return HighlightResult(
        highlights=(),
        error=(
            f'Model returned {len(highlights)} usable highlight(s); Merchant'
            f' Center requires at least {MIN_HIGHLIGHTS}'
        ),
    )
  return HighlightResult(highlights=highlights)


def validate_paths(input_path: str, output_path: str) -> None:
  """Checks the feed paths before any work starts.

  Args:
    input_path: Path to the input feed.
    output_path: Path the output feed will be written to.

  Raises:
    FileNotFoundError: If the input feed does not exist.
    ValueError: If the output path is the input feed, which would be erased
      before it is read.
  """
  if not os.path.isfile(input_path):
    raise FileNotFoundError(
        f'Input feed {input_path!r} does not exist or is not a regular file.'
    )
  if os.path.realpath(input_path) == os.path.realpath(output_path):
    raise ValueError(
        f'--output_file {output_path!r} is the input feed. Choose a different'
        ' output path; the input would be erased before it is read.'
    )


def process_feed(
    input_path: str,
    output_path: str,
    generator: TextGenerator,
    num_rows: int = 0,
    http_get: HttpGet = requests.get,
) -> FeedSummary:
  """Streams a feed through the model, writing each product as it finishes.

  Products that fail are logged and written with an empty
  `product_highlight` value, so the output never contains error text as
  highlights. An existing `product_highlight` column is overwritten.

  Args:
    input_path: Path to the input feed (CSV or TSV).
    output_path: Path to write the enriched TSV feed to.
    generator: Sends chat messages to the model.
    num_rows: Number of products to process. 0 processes all products.
    http_get: Function used to download product images.

  Returns:
    Counts of processed, failed and skipped products.

  Raises:
    FileNotFoundError: If the input feed does not exist.
    ValueError: If the paths are invalid, or the feed is malformed in a way
      that cannot be skipped safely.
  """
  validate_paths(input_path, output_path)
  processed_count = 0
  failed_count = 0
  # `utf-8-sig` drops the byte-order mark that spreadsheet exports often add.
  with (
      open(input_path, encoding='utf-8-sig', newline='') as input_file,
      open(output_path, 'w', encoding='utf-8', newline='') as output_file,
  ):
    reader = FeedReader(input_file, input_path)
    columns = list(reader.columns)
    if HIGHLIGHT_COLUMN not in columns:
      columns.append(HIGHLIGHT_COLUMN)
    writer = TsvWriter(output_file, columns)
    # Limiting the rows up front, rather than breaking out of the loop,
    # lets the progress bar count the last product too.
    products = itertools.islice(reader, num_rows) if num_rows else reader
    for product in tqdm.tqdm(
        products, total=num_rows or None, desc='Generating highlights'
    ):
      result = generate_highlights(product, generator, http_get)
      if result.error:
        failed_count += 1
        logging.warning(
            'No highlights for product %s: %s',
            product.get('id') or '(no id)',
            result.error,
        )
      writer.write_row(
          {**product, HIGHLIGHT_COLUMN: format_highlights(result.highlights)}
      )
      processed_count += 1
  return FeedSummary(
      processed_count=processed_count,
      failed_count=failed_count,
      skipped_line_count=reader.skipped_line_count,
  )


def resolve_model_path(
    model_path: str,
    downloader: Callable[[str], str] = kagglehub.model_download,
) -> str:
  """Returns the directory to load the Gemma model from.

  Args:
    model_path: User-supplied local directory. If empty, the model is downloaded
      from Kaggle instead.
    downloader: Function that downloads a Kaggle model handle and returns its
      local path.

  Returns:
    Path to a local directory containing the model weights.

  Raises:
    FileNotFoundError: If `model_path` is set but is not a directory.
  """
  if not model_path:
    return downloader(KAGGLE_MODEL_HANDLE)
  if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f'--model_path {model_path!r} is not a directory. Pass a local'
        ' directory containing the Gemma weights, or leave it empty to'
        ' download the model from Kaggle.'
    )
  return model_path


def load_generator(model_path: str) -> TextGenerator:
  """Loads the Gemma model as a text generator.

  Args:
    model_path: Local directory containing the model weights.

  Returns:
    A function that sends chat messages to the model.
  """
  pipe = transformers.pipeline(
      task='image-text-to-text',
      model=model_path,
      device_map='auto',
      dtype='auto',
  )

  def _generate(messages: list[dict[str, Any]]) -> Any:
    return pipe(
        messages,
        return_full_text=False,
        generate_kwargs={'max_new_tokens': MAX_NEW_TOKENS},
    )

  return _generate


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Checked here rather than with `flags.mark_flags_as_required`, which has no
  # effect on flags with a non-None default, and which would make the flags
  # mandatory for anything that imports this module, such as tests.
  if not _INPUT_FILE.value or not _OUTPUT_FILE.value:
    raise app.UsageError('--input_file and --output_file are required.')

  # Check the paths first, so a typo fails before the model is downloaded or
  # loaded.
  validate_paths(_INPUT_FILE.value, _OUTPUT_FILE.value)
  generator = load_generator(resolve_model_path(_MODEL_PATH.value))
  summary = process_feed(
      _INPUT_FILE.value, _OUTPUT_FILE.value, generator, _NUM_ROWS.value
  )
  logging.info(
      'Wrote %d product(s) to %s.', summary.processed_count, _OUTPUT_FILE.value
  )
  if summary.failed_count:
    logging.warning(
        '%d product(s) have no highlights; see warnings above.',
        summary.failed_count,
    )
  if summary.skipped_line_count:
    logging.warning(
        'Skipped %d malformed line(s) in %s; see warnings above.',
        summary.skipped_line_count,
        _INPUT_FILE.value,
    )


def run() -> None:
  """Entry point for the script and the `highlight-gen` console command."""
  app.run(main)


if __name__ == '__main__':
  run()
