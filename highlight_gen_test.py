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
from PIL import Image
import requests

import highlight_gen

_IMAGE_URL = 'https://cdn.example.com/shoe.jpg'


def _png_bytes() -> bytes:
  buffer = io.BytesIO()
  Image.new('RGB', (4, 4), color='red').save(buffer, format='PNG')
  return buffer.getvalue()


def _product(**overrides: str) -> dict[str, str]:
  product = {
      'id': 'sku-1',
      'title': 'Trail running shoe',
      'description': 'Lightweight shoe for off-road running.',
      'color': 'Red',
      'material': 'Mesh',
      'size': '42',
      'image_link': _IMAGE_URL,
      'additional_image_link': '',
  }
  product.update(overrides)
  return product


class _FakeResponse(requests.Response):
  """An HTTP response with a fixed status code and body.

  Attributes:
    status_code: The HTTP status code.
    reason: The reason phrase for the status code.
    url: The URL that was requested.
  """

  def __init__(self, status_code: int, body: bytes = b''):
    super().__init__()
    self.status_code = status_code
    self.reason = 'OK' if status_code < 400 else 'Error'
    self.url = _IMAGE_URL
    self._body = body

  @property
  def content(self) -> bytes:
    return self._body


class _FakeHttpGet:
  """Stands in for `requests.get`, returning a fixed response.

  Attributes:
    calls: The arguments passed to the fake HTTP GET function.
  """

  def __init__(self, response: requests.Response):
    self._response = response
    self.calls = []

  def __call__(self, url: str, **kwargs) -> requests.Response:
    self.calls.append((url, kwargs))
    return self._response


class _FakeGenerator:
  """Stands in for the Gemma pipeline, returning fixed text.

  Attributes:
    calls: The messages passed to the generator.
  """

  def __init__(self, text: str = '', error: Exception | None = None):
    self._text = text
    self._error = error
    self.calls = []

  def __call__(self, messages):
    self.calls.append(messages)
    if self._error is not None:
      raise self._error
    return [{'generated_text': self._text}]


def _image_get() -> _FakeHttpGet:
  return _FakeHttpGet(_FakeResponse(200, _png_bytes()))


class SelectImageUrlTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('image_link', _IMAGE_URL, '', _IMAGE_URL),
      (
          'falls_back_to_additional',
          '',
          'https://cdn.example.com/b.jpg',
          'https://cdn.example.com/b.jpg',
      ),
      (
          'first_of_several',
          'https://cdn.example.com/a.jpg,https://cdn.example.com/c.jpg',
          '',
          'https://cdn.example.com/a.jpg',
      ),
      ('ignores_non_http', 'ftp://cdn.example.com/a.jpg', '', ''),
      ('no_links', '', '', ''),
      (
          'cloudinary_commas_kept',
          'https://res.cloudinary.com/demo/upload/w_800,h_800,c_pad/a.jpg',
          '',
          'https://res.cloudinary.com/demo/upload/w_800,h_800,c_pad/a.jpg',
      ),
      (
          'cloudflare_commas_kept',
          'https://cdn.example.com/cdn-cgi/image/width=800,quality=85/a.jpg',
          '',
          'https://cdn.example.com/cdn-cgi/image/width=800,quality=85/a.jpg',
      ),
      (
          'comma_url_then_second_url',
          'https://img.example/w_800,h_800/a.jpg,https://img.example/c.jpg',
          '',
          'https://img.example/w_800,h_800/a.jpg',
      ),
      (
          'comma_and_space_separated',
          'https://cdn.example.com/a.jpg, https://cdn.example.com/c.jpg',
          '',
          'https://cdn.example.com/a.jpg',
      ),
      (
          'uppercase_scheme',
          'HTTPS://cdn.example.com/a.jpg',
          '',
          'HTTPS://cdn.example.com/a.jpg',
      ),
      (
          'non_http_then_http',
          'ftp://cdn.example.com/a.jpg,https://cdn.example.com/b.jpg',
          '',
          'https://cdn.example.com/b.jpg',
      ),
      (
          'space_before_comma',
          'ftp://cdn.example.com/a.jpg ,https://cdn.example.com/b.jpg',
          '',
          'https://cdn.example.com/b.jpg',
      ),
  )
  def test_select_image_url(self, image_link, additional, expected):
    product = _product(image_link=image_link, additional_image_link=additional)

    url = highlight_gen.select_image_url(product)

    self.assertEqual(url, expected)


class FetchImageTest(absltest.TestCase):

  def test_sends_honest_user_agent(self):
    http_get = _image_get()

    highlight_gen.fetch_image(_IMAGE_URL, http_get)

    _, kwargs = http_get.calls[0]
    self.assertEqual(
        kwargs['headers']['User-Agent'],
        'HighlightGen/0.1.0'
        ' (+https://github.com/google-marketing-solutions/highlight_gen)',
    )

  def test_returns_rgb_image(self):
    image = highlight_gen.fetch_image(_IMAGE_URL, _image_get())

    self.assertEqual(image.mode, 'RGB')

  def test_forbidden_raises_without_retrying(self):
    http_get = _FakeHttpGet(_FakeResponse(403))

    with self.assertRaisesRegex(highlight_gen.ImageFetchError, 'download'):
      highlight_gen.fetch_image(_IMAGE_URL, http_get)
    self.assertLen(http_get.calls, 1)

  def test_non_image_body_raises(self):
    http_get = _FakeHttpGet(_FakeResponse(200, b'<html>not an image</html>'))

    with self.assertRaisesRegex(highlight_gen.ImageFetchError, 'decode'):
      highlight_gen.fetch_image(_IMAGE_URL, http_get)


class BuildMessagesTest(absltest.TestCase):

  def test_includes_image_and_non_empty_details(self):
    image = Image.new('RGB', (1, 1))
    product = _product(material='')

    messages = highlight_gen.build_messages(product, image)

    content = messages[0]['content']
    self.assertIs(content[0]['image'], image)
    details = ' '.join(
        part['text'] for part in content if part['type'] == 'text'
    )
    self.assertIn('Title: Trail running shoe', details)
    self.assertIn('Color: Red', details)
    self.assertNotIn('Material', details)

  def test_omits_blank_title(self):
    product = _product(title='  ')

    messages = highlight_gen.build_messages(product, Image.new('RGB', (1, 1)))

    content = messages[0]['content']
    details = ' '.join(
        part['text'] for part in content if part['type'] == 'text'
    )
    self.assertNotIn('Title', details)
    self.assertIn('Color: Red', details)


class CleanModelOutputTest(absltest.TestCase):

  def test_removes_markdown_bold_and_preamble(self):
    raw = 'Here are 3 highlights: **Breathable mesh**'

    cleaned = highlight_gen.clean_model_output(raw)

    self.assertEqual(cleaned, 'Breathable mesh')


class ParseHighlightsTest(parameterized.TestCase):

  def test_splits_and_strips(self):
    highlights = highlight_gen.parse_highlights(
        ' Breathable mesh , Grippy sole.,- Lightweight'
    )

    self.assertEqual(
        highlights, ['Breathable mesh', 'Grippy sole', 'Lightweight']
    )

  @parameterized.named_parameters(
      ('dash', '- Grippy sole', 'Grippy sole'),
      ('asterisk', '* Grippy sole', 'Grippy sole'),
      ('bullet_without_space', '•Grippy sole', 'Grippy sole'),
      ('numbered_with_dot', '1. Grippy sole', 'Grippy sole'),
      ('numbered_with_bracket', '2) Grippy sole', 'Grippy sole'),
  )
  def test_removes_list_markers(self, value, expected):
    highlights = highlight_gen.parse_highlights(f'{value},Lightweight')

    self.assertEqual(highlights, [expected, 'Lightweight'])

  @parameterized.named_parameters(
      ('decimal', '1.5mm thick steel'),
      ('negative_number', '-20°C rated'),
      ('leading_number', '3 column design'),
  )
  def test_keeps_values_that_start_like_list_markers(self, value):
    highlights = highlight_gen.parse_highlights(f'{value},Lightweight')

    self.assertEqual(highlights, [value, 'Lightweight'])

  def test_drops_empty_duplicate_and_overlong_values(self):
    overlong = 'x' * 151

    highlights = highlight_gen.parse_highlights(
        f'Grippy sole,,grippy sole,{overlong},Lightweight'
    )

    self.assertEqual(highlights, ['Grippy sole', 'Lightweight'])

  def test_keeps_at_most_max_highlights(self):
    text = ','.join(f'Highlight {i}' for i in range(10))

    highlights = highlight_gen.parse_highlights(text)

    self.assertLen(highlights, 5)


class FormatHighlightsTest(absltest.TestCase):

  def test_joins_with_commas(self):
    formatted = highlight_gen.format_highlights(['Grippy sole', 'Lightweight'])

    self.assertEqual(formatted, 'Grippy sole,Lightweight')

  def test_quotes_values_with_commas_or_colons(self):
    formatted = highlight_gen.format_highlights(
        ['Drop: 8 mm', 'Light, breathable']
    )

    self.assertEqual(formatted, '"Drop: 8 mm","Light, breathable"')

  def test_quotes_values_with_inner_quotes_and_doubles_them(self):
    formatted = highlight_gen.format_highlights(
        ['Fit: "true to size"', '"Waterproof" rating']
    )

    self.assertEqual(
        formatted, '"Fit: ""true to size""","""Waterproof"" rating"'
    )

  def test_empty_list_gives_empty_string(self):
    self.assertEqual(highlight_gen.format_highlights([]), '')


class GenerateHighlightsTest(absltest.TestCase):

  def test_success(self):
    generator = _FakeGenerator('Breathable mesh, Grippy sole, Lightweight')

    result = highlight_gen.generate_highlights(
        _product(), generator, _image_get()
    )

    self.assertEqual(result.error, '')
    self.assertEqual(
        result.highlights, ['Breathable mesh', 'Grippy sole', 'Lightweight']
    )

  def test_no_image_link_skips_model(self):
    generator = _FakeGenerator('unused')

    result = highlight_gen.generate_highlights(
        _product(image_link=''), generator, _image_get()
    )

    self.assertEqual(result.error, 'No image link provided')
    self.assertEmpty(result.highlights)
    self.assertEmpty(generator.calls)

  def test_forbidden_image_is_reported(self):
    generator = _FakeGenerator('unused')

    result = highlight_gen.generate_highlights(
        _product(), generator, _FakeHttpGet(_FakeResponse(403))
    )

    self.assertIn('Could not download image', result.error)
    self.assertEmpty(result.highlights)
    self.assertEmpty(generator.calls)

  def test_too_few_highlights_is_reported(self):
    generator = _FakeGenerator('Breathable mesh')

    result = highlight_gen.generate_highlights(
        _product(), generator, _image_get()
    )

    self.assertIn('at least 2', result.error)
    self.assertEmpty(result.highlights)

  def test_model_error_is_reported(self):
    generator = _FakeGenerator(error=RuntimeError('out of memory'))

    result = highlight_gen.generate_highlights(
        _product(), generator, _image_get()
    )

    self.assertIn('out of memory', result.error)
    self.assertEmpty(result.highlights)


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
