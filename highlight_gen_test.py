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

import io

from absl.testing import absltest
from absl.testing import parameterized

import highlight_gen


def _read_feed(contents: str) -> tuple[list[str], list[dict[str, str]], int]:
  reader = highlight_gen.FeedReader(io.StringIO(contents), 'feed.csv')
  rows = list(reader)
  return reader.columns, rows, reader.skipped_line_count


class FeedReaderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('csv', 'ID,Title\nsku-1,Shoe\nsku-2,Boot\n'),
      ('tsv', 'ID\tTitle\nsku-1\tShoe\nsku-2\tBoot\n'),
  )
  def test_detects_delimiter_and_lowercases_columns(self, contents):
    columns, rows, _ = _read_feed(contents)

    self.assertEqual(columns, ['id', 'title'])
    self.assertEqual(
        rows,
        [{'id': 'sku-1', 'title': 'Shoe'}, {'id': 'sku-2', 'title': 'Boot'}],
    )

  def test_quoted_values_with_delimiters_are_kept_whole(self):
    _, rows, _ = _read_feed('id,description\nsku-1,"Light, breathable"\n')

    self.assertEqual(rows[0]['description'], 'Light, breathable')

  def test_skips_lines_with_extra_fields_without_shifting_columns(self):
    _, rows, skipped = _read_feed(
        'id,title\na,Shoe\nb,Boot,extra,fields\nc,Sock\n'
    )

    self.assertEqual(skipped, 1)
    self.assertEqual(
        rows, [{'id': 'a', 'title': 'Shoe'}, {'id': 'c', 'title': 'Sock'}]
    )

  def test_pads_short_lines_and_ignores_blank_lines(self):
    _, rows, skipped = _read_feed('id,title,size\na,Shoe\n\n')

    self.assertEqual(rows, [{'id': 'a', 'title': 'Shoe', 'size': ''}])
    self.assertEqual(skipped, 0)

  def test_header_without_delimiter_raises(self):
    with self.assertRaisesRegex(ValueError, 'Could not detect the delimiter'):
      _read_feed('title\nShoe\n')


class TsvWriterTest(absltest.TestCase):

  def test_writes_unquoted_tsv_and_sanitises_values(self):
    output = io.StringIO()
    writer = highlight_gen.TsvWriter(
        output, ['id', 'description', 'product_highlight']
    )

    writer.write_row({
        'id': 'sku-1',
        'description': 'Line one\nline\ttwo',
        'product_highlight': '"Drop: 8 mm",Lightweight',
    })

    self.assertEqual(
        output.getvalue().splitlines(),
        [
            'id\tdescription\tproduct_highlight',
            'sku-1\tLine one line two\t"Drop: 8 mm",Lightweight',
        ],
    )

  def test_missing_columns_are_written_empty(self):
    output = io.StringIO()
    writer = highlight_gen.TsvWriter(output, ['id', 'title'])

    writer.write_row({'id': 'sku-1'})

    self.assertEqual(output.getvalue().splitlines()[1], 'sku-1\t')


if __name__ == '__main__':
  absltest.main()
