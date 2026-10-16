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

from collections.abc import Iterator, Mapping, Sequence
import csv
import re
from typing import TextIO

from absl import logging

# Candidate feed delimiters, detected from the header line.
_FEED_DELIMITERS = ',\t;|'

# Tabs and line breaks inside values would break the TSV structure.
_LINE_BREAKS_AND_TABS = re.compile(r'[\t\r\n]+')


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
