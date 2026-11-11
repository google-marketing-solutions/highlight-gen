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
import copy
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
    'Path to write the enriched feed to (always written as TSV). If empty, it'
    ' is written next to the input feed as <input name>_with_highlights.tsv.',
)
_OVERWRITE = flags.DEFINE_bool(
    'overwrite',
    False,
    'Replace the output file if it already exists and is not empty. Without'
    ' this flag, HighlightGen stops rather than overwrite it.',
)
_RESUME = flags.DEFINE_bool(
    'resume',
    False,
    'Continue an interrupted run: keep the products already in the output file'
    ' and process only the rest. Products that failed before are retried.',
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

# transformers' own defaults for the sampling settings. They have no effect
# under greedy decoding. Unlike None, which generate() replaces with the
# model's defaults, they don't trigger warnings about unused settings.
_NEUTRAL_SAMPLING_SETTINGS = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}

# Stop the run when the model fails on this many products in a row, since the
# cause (e.g. GPU out of memory) is then almost certainly systemic.
MAX_CONSECUTIVE_MODEL_FAILURES = 10

# The csv module's default field size limit (128 KiB) is too small for some
# feeds, e.g. descriptions containing HTML.
_MAX_FEED_FIELD_BYTES = 10 * 1024 * 1024
# Block size for finding the last line break of an output file when resuming.
_BACKWARD_SCAN_BLOCK_BYTES = 64 * 1024

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


class TooManyModelFailuresError(RuntimeError):
  """Raised when the model fails on too many products in a row."""


@dataclasses.dataclass(frozen=True)
class HighlightResult:
  """The outcome of generating highlights for one product.

  Attributes:
    highlights: The validated highlights. Empty if generation failed.
    error: Why generation failed, or an empty string on success.
    model_failed: Whether the failure came from the model call itself, as
      opposed to the product's data (e.g. a missing or unreachable image).
  """

  highlights: Sequence[str]
  error: str = ''
  model_failed: bool = False


@dataclasses.dataclass(frozen=True)
class FeedSummary:
  """Counts from processing a feed.

  Attributes:
    processed_count: Number of products processed.
    written_count: Number of processed products written with highlights.
    failed_count: Number of processed products left out of the output because no
      valid highlights could be generated.
    skipped_line_count: Number of malformed input lines that were skipped.
    resumed_count: Number of products kept from an earlier run's output, which
      were not processed again.
  """

  processed_count: int
  written_count: int
  failed_count: int
  skipped_line_count: int
  resumed_count: int


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
  missing trailing values are treated as empty. Blank lines are ignored, and
  NULL bytes are removed.

  Quoting is parsed strictly. A broken quoted value (e.g. a missing closing
  quote) raises an error rather than being skipped, because the parser cannot
  tell where the broken record ends: it would otherwise silently absorb the
  rest of the file into one value, or turn it into made-up products.

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
      ValueError: If the feed's delimiter cannot be detected, or its header is
        malformed or not valid UTF-8.
    """
    self._source_name = source_name
    try:
      header_line = feed_file.readline()
    except UnicodeDecodeError as e:
      raise self._encoding_error(1, e) from e
    delimiter = _detect_delimiter(header_line, source_name)
    feed_file.seek(0)
    # Raises the process-wide limit, but never lowers it, so this is harmless
    # when the module is used as a library.
    csv.field_size_limit(max(csv.field_size_limit(), _MAX_FEED_FIELD_BYTES))
    self._reader = csv.reader(
        (line.replace('\0', '') for line in feed_file),
        delimiter=delimiter,
        strict=True,
    )
    try:
      header = next(self._reader)
    except csv.Error as e:
      raise self._malformed_record_error(1, e) from e
    self.columns = [column.strip().lower() for column in header]
    self.skipped_line_count = 0

  def __iter__(self) -> Iterator[dict[str, str]]:
    """Yields each product as a mapping from column name to value.

    Yields:
      One dict per product, keyed by the lower-cased column names. Missing
      trailing values are empty strings.

    Raises:
      ValueError: If a record is malformed in a way that cannot be skipped
        safely, or the feed is not valid UTF-8.
    """
    while True:
      start_line = self._reader.line_num + 1
      try:
        fields = next(self._reader)
      except StopIteration:
        return
      except csv.Error as e:
        raise self._malformed_record_error(start_line, e) from e
      except UnicodeDecodeError as e:
        raise self._encoding_error(start_line, e) from e

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

  def _malformed_record_error(self, line: int, error: csv.Error) -> ValueError:
    return ValueError(
        f'{self._source_name} has a malformed record starting at line'
        f' {line}: {error}. Fix the record and re-run.'
    )

  def _encoding_error(self, line: int, error: UnicodeDecodeError) -> ValueError:
    # Text is decoded in blocks, so the error position is only approximate.
    return ValueError(
        f'{self._source_name} is not valid UTF-8 near line {line}: {error}.'
        ' Re-save the file as UTF-8 and re-run.'
    )


class TsvWriter:
  """Writes products to a Merchant Center tab-separated file, one at a time.

  Each row is flushed as soon as it is written, so the output keeps every
  finished product if a long run stops part-way through.

  Fields are written without quoting so that the quoting inside
  `product_highlight` values reaches Merchant Center unchanged. Tabs and line
  breaks inside values are replaced with spaces, since they would otherwise
  break the file structure.
  """

  def __init__(
      self,
      output_file: TextIO,
      columns: Sequence[str],
      write_header: bool = True,
  ):
    """Initializes the writer and writes the header row.

    Args:
      output_file: The open output file.
      columns: Column names, in output order.
      write_header: Whether to write the header row. False when appending to a
        file that already has one.
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
    if write_header:
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
    # RuntimeError includes torch.OutOfMemoryError and CUDA errors.
    return HighlightResult(
        highlights=(),
        error=f'Model generation failed: {e}',
        model_failed=True,
    )

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


def _generate_highlights_isolated(
    product: Mapping[str, str],
    generator: TextGenerator,
    http_get: HttpGet,
) -> HighlightResult:
  """Calls `generate_highlights`, turning unexpected errors into a result.

  This is the isolation point that stops one bad product from ending a run
  that may last days. Unexpected errors are logged with their traceback and
  counted as model failures, so a bug that affects every product still stops
  the run (see `MAX_CONSECUTIVE_MODEL_FAILURES`).

  Args:
    product: One feed row.
    generator: Sends chat messages to the model.
    http_get: Function used to download the product image.

  Returns:
    The result of `generate_highlights`, or a failed result.
  """
  try:
    return generate_highlights(product, generator, http_get)
  except Exception as e:
    logging.exception(
        'Unexpected error for product %s', product.get('id') or '(no id)'
    )
    return HighlightResult(
        highlights=(), error=f'Unexpected error: {e!r}', model_failed=True
    )


def default_output_path(input_path: str) -> str:
  """Returns where to write the output when no output path is given.

  Args:
    input_path: Path to the input feed.

  Returns:
    The input path with its extension replaced by `_with_highlights.tsv`, e.g.
    `feed_with_highlights.tsv` for `feed.csv`.
  """
  stem, _ = os.path.splitext(input_path)
  return f'{stem}_with_highlights.tsv'


def failures_path(output_path: str) -> str:
  """Returns the path of the file that lists failed products for an output.

  Args:
    output_path: Path of the enriched output feed.

  Returns:
    The output path with its extension replaced by `_failed.tsv`, e.g.
    `feed_out_failed.tsv` for `feed_out.tsv`.
  """
  stem, _ = os.path.splitext(output_path)
  return f'{stem}_failed.tsv'


def validate_paths(
    input_path: str,
    output_path: str,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
  """Checks the feed paths before any work starts.

  Args:
    input_path: Path to the input feed.
    output_path: Path the output feed will be written to.
    overwrite: Whether an existing, non-empty output may be replaced.
    resume: Whether an existing output will be continued.

  Raises:
    FileNotFoundError: If the input feed does not exist.
    FileExistsError: If the output already exists, is not empty and neither
      `overwrite` nor `resume` is set. A finished run can take days, so it is
      never replaced by accident.
    ValueError: If both `overwrite` and `resume` are set, or the output path,
      or the failures file derived from it, is the input feed, which would be
      erased before it is read.
  """
  if overwrite and resume:
    raise ValueError(
        'overwrite and resume cannot be used together: one replaces the'
        ' existing output and the other continues it.'
    )
  if not os.path.isfile(input_path):
    raise FileNotFoundError(
        f'Input feed {input_path!r} does not exist or is not a regular file.'
    )
  if os.path.realpath(input_path) == os.path.realpath(output_path):
    raise ValueError(
        f'--output_file {output_path!r} is the input feed. Choose a different'
        ' output path; the input would be erased before it is read.'
    )
  if os.path.realpath(input_path) == os.path.realpath(
      failures_path(output_path)
  ):
    raise ValueError(
        f'The failures file for --output_file {output_path!r} would be the'
        ' input feed. Choose a different output path; the input would be'
        ' erased before it is read.'
    )
  if (
      not (overwrite or resume)
      and os.path.isfile(output_path)
      and os.path.getsize(output_path) > 0
  ):
    raise FileExistsError(
        f'Output file {output_path!r} already exists and is not empty. Pass'
        ' --resume to continue it, --overwrite to replace it, or choose a'
        ' different --output_file.'
    )


def _remove_partial_last_line(path: str) -> None:
  """Cuts a file back to its last line break.

  A run that is killed while writing can leave an incomplete last row, which
  would otherwise be kept as a corrupt product.

  Args:
    path: The file to check.
  """
  with open(path, 'rb+') as f:
    end = f.seek(0, os.SEEK_END)
    position = end
    # Scan backwards in blocks, so huge files are not read into memory.
    while position > 0:
      block_start = max(0, position - _BACKWARD_SCAN_BLOCK_BYTES)
      f.seek(block_start)
      block = f.read(position - block_start)
      line_break = block.rfind(b'\n')
      if line_break != -1:
        keep = block_start + line_break + 1
        break
      position = block_start
    else:
      keep = 0
    if keep < end:
      f.truncate(keep)
      logging.warning(
          'Removed an incomplete last line from %s, left by an interrupted'
          ' run.',
          path,
      )


def read_written_ids(output_path: str, columns: Sequence[str]) -> set[str]:
  """Returns the ids of products already in an output, to resume a run.

  An incomplete last line, left by a run that was killed while writing, is
  removed from the file first.

  Args:
    output_path: Path of the output written by the earlier run.
    columns: The columns the output is expected to have, in order.

  Returns:
    The non-empty ids already written. Empty if there is nothing to resume, in
    which case the output should be written from the start.

  Raises:
    ValueError: If the output's header does not match `columns`, e.g. because
      it was written for a different input feed, or has no `id` column to
      tell which products were already written.
  """
  if not os.path.isfile(output_path):
    logging.info('Nothing to resume: %s does not exist yet.', output_path)
    return set()
  _remove_partial_last_line(output_path)
  with open(output_path, encoding='utf-8', newline='') as output_file:
    rows = csv.reader(output_file, delimiter='\t', quoting=csv.QUOTE_NONE)
    header = next(rows, None)
    if header is None:
      logging.info('Nothing to resume: %s is empty.', output_path)
      return set()
    if header != list(columns):
      raise ValueError(
          f'Cannot resume {output_path!r}: its columns {header} do not match'
          f' the columns expected for this input feed, {list(columns)}. Check'
          ' that --input_file is the feed the output was written from.'
      )
    if 'id' not in header:
      raise ValueError(
          f'Cannot resume {output_path!r}: it has no id column, which is'
          ' needed to tell which products were already written.'
      )
    id_index = header.index('id')
    ids = (row[id_index].strip() for row in rows if len(row) > id_index)
    # A blank id cannot identify a product, so it never marks one as written.
    return {product_id for product_id in ids if product_id}


def _claim_product_id(product: Mapping[str, str], seen_ids: set[str]) -> str:
  """Records a product's id as seen, or says why the product must be skipped.

  Merchant Center needs an `id` to match each row to a product, and a feed
  with the same id twice is ambiguous, so only the first row with each id is
  processed.

  Args:
    product: One feed row.
    seen_ids: Ids of the products already processed in this run. The product's
      id is added if it is valid.

  Returns:
    Why the product must be skipped, or an empty string if it can be processed.
  """
  product_id = product.get('id', '').strip()
  if not product_id:
    return 'Missing id'
  if product_id in seen_ids:
    return 'Duplicate id; only the first row with this id is used'
  seen_ids.add(product_id)
  return ''


def process_feed(
    input_path: str,
    output_path: str,
    generator: TextGenerator,
    num_rows: int = 0,
    http_get: HttpGet = requests.get,
    overwrite: bool = False,
    resume: bool = False,
) -> FeedSummary:
  """Streams a feed through the model, writing each product as it finishes.

  Only products with valid highlights are written. Products that fail are
  logged and left out of the output, so it never contains empty or error
  values as highlights. An existing `product_highlight` column is overwritten.
  Products without an `id`, and repeats of an id already seen, are skipped
  without calling the model. Every product left out is also listed, with the
  reason, in the file given by `failures_path`, which is rewritten on each run.

  If the model fails on `MAX_CONSECUTIVE_MODEL_FAILURES` products in a row, the
  run stops, since the cause (e.g. GPU out of memory) is then almost certainly
  systemic. Failures caused by a product's data, such as an unreachable image,
  do not count towards this limit and reset it. Skipped ids leave it unchanged.

  With `resume`, products already in the output are kept and not processed
  again, and new products are appended. Products that failed in the earlier
  run are not in the output, so they are retried.

  Args:
    input_path: Path to the input feed (CSV or TSV).
    output_path: Path to write the enriched TSV feed to.
    generator: Sends chat messages to the model.
    num_rows: Number of products to process. 0 processes all products.
    http_get: Function used to download product images.
    overwrite: Whether an existing, non-empty output may be replaced.
    resume: Whether to continue an existing output instead of replacing it.

  Returns:
    Counts of processed, written, failed, skipped and resumed products.

  Raises:
    FileNotFoundError: If the input feed does not exist.
    FileExistsError: If the output already exists, is not empty and neither
      `overwrite` nor `resume` is set.
    ValueError: If the paths are invalid, the feed is malformed in a way that
      cannot be skipped safely, or the output to resume has different columns.
    TooManyModelFailuresError: If the model fails on too many products in a
      row.
  """
  validate_paths(input_path, output_path, overwrite, resume)
  processed_count = 0
  failed_count = 0
  resumed_count = 0
  consecutive_model_failures = 0
  seen_ids = set()
  # `utf-8-sig` drops the byte-order mark that spreadsheet exports often add.
  with open(input_path, encoding='utf-8-sig', newline='') as input_file:
    reader = FeedReader(input_file, input_path)
    columns = list(reader.columns)
    if HIGHLIGHT_COLUMN not in columns:
      columns.append(HIGHLIGHT_COLUMN)
    written_ids = read_written_ids(output_path, columns) if resume else set()
    with (
        open(
            output_path,
            'a' if written_ids else 'w',
            encoding='utf-8',
            newline='',
        ) as output_file,
        open(
            failures_path(output_path), 'w', encoding='utf-8', newline=''
        ) as failures_file,
    ):
      writer = TsvWriter(output_file, columns, write_header=not written_ids)
      failures_writer = TsvWriter(failures_file, ['id', 'reason'])
      # Limiting the rows up front, rather than breaking out of the loop,
      # lets the progress bar count the last product too. When resuming,
      # this keeps --num_rows the total for the whole job.
      products = itertools.islice(reader, num_rows) if num_rows else reader
      for product in tqdm.tqdm(
          products, total=num_rows or None, desc='Generating highlights'
      ):
        product_id = product.get('id', '').strip()
        if product_id in written_ids and product_id not in seen_ids:
          seen_ids.add(product_id)
          resumed_count += 1
          continue
        id_error = _claim_product_id(product, seen_ids)
        if id_error:
          result = HighlightResult(highlights=(), error=id_error)
        else:
          result = _generate_highlights_isolated(product, generator, http_get)
        processed_count += 1
        if not result.error:
          consecutive_model_failures = 0
          writer.write_row({
              **product,
              HIGHLIGHT_COLUMN: format_highlights(result.highlights),
          })
          continue
        failed_count += 1
        logging.warning(
            'No highlights for product %s: %s',
            product.get('id') or '(no id)',
            result.error,
        )
        failures_writer.write_row(
            {'id': product.get('id', ''), 'reason': result.error}
        )
        if id_error:
          # The model was not called, so this says nothing about whether it
          # works: the count of consecutive model failures is left unchanged.
          continue
        if not result.model_failed:
          consecutive_model_failures = 0
          continue
        consecutive_model_failures += 1
        if consecutive_model_failures >= MAX_CONSECUTIVE_MODEL_FAILURES:
          raise TooManyModelFailuresError(
              f'The model failed on {consecutive_model_failures} products in a'
              f' row; last error: {result.error}. Fix the cause and re-run'
              ' with --resume.'
          )
  return FeedSummary(
      processed_count=processed_count,
      written_count=processed_count - failed_count,
      failed_count=failed_count,
      skipped_line_count=reader.skipped_line_count,
      resumed_count=resumed_count,
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


def build_generation_config(
    base: transformers.GenerationConfig,
) -> transformers.GenerationConfig:
  """Returns the generation settings for highlights, based on the model's own.

  All settings are kept in one config object, rather than partly passed as
  separate arguments. Passing both made transformers log a warning on every
  product, which would flood the log on large feeds.

  Args:
    base: The model's default generation config. It is not modified.

  Returns:
    A copy of `base` set up for greedy decoding of up to `MAX_NEW_TOKENS`.
  """
  config = copy.deepcopy(base)
  config.max_new_tokens = MAX_NEW_TOKENS
  # Unset the library default (20 tokens), so that only max_new_tokens limits
  # the output length.
  config.max_length = None
  # Greedy decoding: the same feed always produces the same highlights, which
  # makes runs reproducible and resumed runs consistent.
  config.do_sample = False
  config.update(**_NEUTRAL_SAMPLING_SETTINGS)
  return config


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
  generation_config = build_generation_config(pipe.generation_config)

  def _generate(messages: list[dict[str, Any]]) -> Any:
    # The pipeline accepts chat messages (a list of dicts) as its first
    # argument, but its `__call__` overloads only declare image and text inputs,
    # not the chat format, so the type checker cannot match this call.
    # pyrefly: ignore[no-matching-overload]
    return pipe(
        messages,
        return_full_text=False,
        generate_kwargs={'generation_config': generation_config},
    )

  return _generate


def _log_summary(
    summary: FeedSummary, input_path: str, output_path: str
) -> None:
  """Logs the outcome of a run, pointing to the files to look at next."""
  if summary.resumed_count:
    logging.info(
        'Kept %d product(s) already in %s from the earlier run.',
        summary.resumed_count,
        output_path,
    )
  logging.info('Wrote %d product(s) to %s.', summary.written_count, output_path)
  if summary.failed_count:
    logging.warning(
        '%d product(s) were left out because no valid highlights could be'
        ' generated; see %s for the reasons.',
        summary.failed_count,
        failures_path(output_path),
    )
  if summary.skipped_line_count:
    logging.warning(
        'Skipped %d malformed line(s) in %s; see warnings above.',
        summary.skipped_line_count,
        input_path,
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Checked here rather than with `flags.mark_flags_as_required`, which has no
  # effect on flags with a non-None default, and which would make the flag
  # mandatory for anything that imports this module, such as tests.
  if not _INPUT_FILE.value:
    raise app.UsageError('--input_file is required.')
  if _OVERWRITE.value and _RESUME.value:
    raise app.UsageError('--overwrite and --resume cannot be used together.')
  input_path = _INPUT_FILE.value
  output_path = _OUTPUT_FILE.value or default_output_path(input_path)

  # Check the paths first, so a typo fails before the model is downloaded or
  # loaded.
  validate_paths(input_path, output_path, _OVERWRITE.value, _RESUME.value)
  logging.info('Writing highlights to %s.', output_path)
  generator = load_generator(resolve_model_path(_MODEL_PATH.value))
  summary = process_feed(
      input_path,
      output_path,
      generator,
      _NUM_ROWS.value,
      overwrite=_OVERWRITE.value,
      resume=_RESUME.value,
  )
  _log_summary(summary, input_path, output_path)


def run() -> None:
  """Entry point for the script and the `highlight-gen` console command."""
  app.run(main)


if __name__ == '__main__':
  run()
