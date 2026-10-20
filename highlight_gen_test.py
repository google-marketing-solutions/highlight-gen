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

from collections.abc import Iterator
import io
import os
from unittest import mock

from absl import app
from absl.testing import absltest
from absl.testing import flagsaver
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
  """An HTTP response with a fixed status code, body and headers.

  Attributes:
    status_code: The HTTP status code.
    reason: The reason phrase for the status code.
    url: The URL that was requested.
  """

  def __init__(
      self,
      status_code: int,
      body: bytes = b'',
      headers: dict[str, str] | None = None,
  ):
    super().__init__()
    self.status_code = status_code
    self.reason = 'OK' if status_code < 400 else 'Error'
    self.url = _IMAGE_URL
    self.headers.update(headers or {})
    self._body = body

  @property
  def content(self) -> bytes:
    return self._body

  def iter_content(
      self, chunk_size: int | None = 1, decode_unicode: bool = False
  ) -> Iterator[bytes]:
    del decode_unicode  # Unused.
    step = chunk_size or len(self._body) or 1
    for start in range(0, len(self._body), step):
      yield self._body[start : start + step]

  def close(self) -> None:
    pass  # There is no network connection to release.


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


class _FakeDownloader:
  """Stands in for `kagglehub.model_download`, recording requested handles."""

  def __init__(self, model_path: str = '/unused/model/path'):
    self._model_path = model_path
    self.handles = []

  def __call__(self, handle: str) -> str:
    self.handles.append(handle)
    return self._model_path


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

  def test_streams_the_download(self):
    http_get = _image_get()

    highlight_gen.fetch_image(_IMAGE_URL, http_get)

    _, kwargs = http_get.calls[0]
    self.assertTrue(kwargs['stream'])

  def test_declared_size_over_limit_raises(self):
    http_get = _FakeHttpGet(
        _FakeResponse(
            200, _png_bytes(), headers={'Content-Length': str(20 * 1024**2 + 1)}
        )
    )

    with self.assertRaisesRegex(
        highlight_gen.ImageFetchError, 'larger than the 20 MB limit'
    ):
      highlight_gen.fetch_image(_IMAGE_URL, http_get)

  def test_body_over_limit_raises_when_size_is_not_declared(self):
    http_get = _FakeHttpGet(_FakeResponse(200, b'\0' * (20 * 1024**2 + 1)))

    with self.assertRaisesRegex(
        highlight_gen.ImageFetchError, 'larger than the 20 MB limit'
    ):
      highlight_gen.fetch_image(_IMAGE_URL, http_get)

  def test_decompression_bomb_raises(self):
    # PIL refuses images over twice MAX_IMAGE_PIXELS; the test image has 16.
    self.enter_context(mock.patch.object(Image, 'MAX_IMAGE_PIXELS', 4))

    with self.assertRaisesRegex(highlight_gen.ImageFetchError, 'decode'):
      highlight_gen.fetch_image(_IMAGE_URL, _image_get())


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


class _RunStoppedError(Exception):
  """Simulates a run stopping part-way, e.g. a crash or Ctrl+C."""


class _FailOnCallGenerator(_FakeGenerator):
  """Succeeds, then raises an unexpected error on the given call number."""

  def __init__(self, text: str, fail_on_call: int):
    super().__init__(text)
    self._fail_on_call = fail_on_call

  def __call__(self, messages):
    if len(self.calls) + 1 == self._fail_on_call:
      raise _RunStoppedError('Run stopped')
    return super().__call__(messages)


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


class ProcessFeedTest(absltest.TestCase):

  def _run(
      self,
      input_contents: str,
      generator: _FakeGenerator,
      num_rows: int = 0,
  ) -> tuple[highlight_gen.FeedSummary, list[str]]:
    input_path = self.create_tempfile(content=input_contents).full_path
    output_path = os.path.join(self.create_tempdir().full_path, 'out.tsv')
    summary = highlight_gen.process_feed(
        input_path, output_path, generator, num_rows, _image_get()
    )
    with open(output_path, encoding='utf-8') as f:
      return summary, f.read().splitlines()

  def test_writes_highlights_and_leaves_failures_empty(self):
    feed = 'id,image_link\nok,https://cdn.example.com/a.jpg\nbad,\n'
    generator = _FakeGenerator('Breathable mesh, Grippy sole')

    summary, lines = self._run(feed, generator)

    self.assertEqual(
        lines,
        [
            'id\timage_link\tproduct_highlight',
            'ok\thttps://cdn.example.com/a.jpg\tBreathable mesh,Grippy sole',
            'bad\t\t',
        ],
    )
    self.assertEqual(summary.processed_count, 2)
    self.assertEqual(summary.failed_count, 1)

  def test_overwrites_existing_highlight_column(self):
    feed = (
        'id,image_link,product_highlight\n'
        'ok,https://cdn.example.com/a.jpg,Old value\n'
    )
    generator = _FakeGenerator('Breathable mesh, Grippy sole')

    _, lines = self._run(feed, generator)

    self.assertEqual(
        lines,
        [
            'id\timage_link\tproduct_highlight',
            'ok\thttps://cdn.example.com/a.jpg\tBreathable mesh,Grippy sole',
        ],
    )

  def test_num_rows_limits_products(self):
    feed = (
        'id,image_link\n'
        'a,https://cdn.example.com/a.jpg\n'
        'b,https://cdn.example.com/b.jpg\n'
        'c,https://cdn.example.com/c.jpg\n'
    )
    generator = _FakeGenerator('Breathable mesh, Grippy sole')

    summary, lines = self._run(feed, generator, num_rows=2)

    self.assertEqual(summary.processed_count, 2)
    self.assertLen(lines, 3)
    self.assertLen(generator.calls, 2)

  def test_finished_products_are_kept_if_run_stops(self):
    feed = (
        'id,image_link\n'
        'a,https://cdn.example.com/a.jpg\n'
        'b,https://cdn.example.com/b.jpg\n'
    )
    input_path = self.create_tempfile(content=feed).full_path
    output_path = os.path.join(self.create_tempdir().full_path, 'out.tsv')
    generator = _FailOnCallGenerator('Breathable mesh, Grippy sole', 2)

    with self.assertRaisesRegex(_RunStoppedError, 'Run stopped'):
      highlight_gen.process_feed(
          input_path, output_path, generator, 0, _image_get()
      )

    with open(output_path, encoding='utf-8') as f:
      lines = f.read().splitlines()
    self.assertEqual(
        lines,
        [
            'id\timage_link\tproduct_highlight',
            'a\thttps://cdn.example.com/a.jpg\tBreathable mesh,Grippy sole',
        ],
    )

  def test_failure_with_empty_id_is_logged_as_no_id(self):
    feed = 'id,title,image_link\n,Trail shoe,\n'

    with self.assertLogs(level='WARNING') as logs:
      self._run(feed, _FakeGenerator('unused'))

    self.assertIn('No highlights for product (no id):', logs.output[0])

  def test_output_path_equal_to_input_raises_and_keeps_input(self):
    feed = 'id,image_link\na,https://cdn.example.com/a.jpg\n'
    input_path = self.create_tempfile(content=feed).full_path

    with self.assertRaisesRegex(ValueError, 'is the input feed'):
      highlight_gen.process_feed(
          input_path, input_path, _FakeGenerator('unused'), 0, _image_get()
      )

    with open(input_path, encoding='utf-8') as f:
      self.assertEqual(f.read(), feed)

  def test_missing_input_raises(self):
    missing = os.path.join(self.create_tempdir().full_path, 'missing.csv')
    output_path = os.path.join(self.create_tempdir().full_path, 'out.tsv')

    with self.assertRaisesRegex(
        FileNotFoundError, 'does not exist or is not a regular file'
    ):
      highlight_gen.process_feed(
          missing, output_path, _FakeGenerator('unused'), 0, _image_get()
      )


class ResolveModelPathTest(absltest.TestCase):

  def test_empty_path_downloads_from_kaggle(self):
    downloader = _FakeDownloader(model_path='/cache/gemma')

    path = highlight_gen.resolve_model_path('', downloader)

    self.assertEqual(path, '/cache/gemma')
    self.assertEqual(
        downloader.handles, ['google/gemma-4/transformers/gemma-4-e2b-it']
    )

  def test_existing_directory_is_used_without_download(self):
    model_dir = self.create_tempdir().full_path
    downloader = _FakeDownloader()

    path = highlight_gen.resolve_model_path(model_dir, downloader)

    self.assertEqual(path, model_dir)
    self.assertEmpty(downloader.handles)

  def test_missing_directory_raises(self):
    missing = os.path.join(self.create_tempdir().full_path, 'missing')
    downloader = _FakeDownloader()

    with self.assertRaisesRegex(FileNotFoundError, 'not a directory'):
      highlight_gen.resolve_model_path(missing, downloader)
    self.assertEmpty(downloader.handles)


class MainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('missing_input', '', 'out.tsv'),
      ('missing_output', 'feed.csv', ''),
  )
  def test_missing_required_flag_raises_usage_error(
      self, input_file, output_file
  ):
    with flagsaver.flagsaver(input_file=input_file, output_file=output_file):
      with self.assertRaisesRegex(app.UsageError, 'are required'):
        highlight_gen.main(['highlight_gen'])

  def test_extra_positional_argument_raises_usage_error(self):
    with flagsaver.flagsaver(input_file='feed.csv', output_file='out.tsv'):
      with self.assertRaisesRegex(app.UsageError, 'Too many'):
        highlight_gen.main(['highlight_gen', 'unexpected_arg'])


if __name__ == '__main__':
  absltest.main()
