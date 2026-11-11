"""Structural checks for the public undergraduate presentation."""
from html.parser import HTMLParser
from pathlib import Path
import unittest
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / 'docs/pregrado/index.html'


class DeckParser(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.slides = []
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article' and 'slide' in attrs.get('class', '').split():
            self.slides.append(attrs)
        if tag == 'button' and 'rail-title' in attrs.get('class', '').split():
            self.links.append(int(attrs['data-first']))


class UndergraduateDeckTests(unittest.TestCase):
    def setUp(self):
        self.text = DECK.read_text()
        self.deck = DeckParser(self.text)

    def test_removed_module_and_reachable_remaining_modules(self):
        self.assertNotIn('Datos: de crudos a utilizables', self.text)
        self.assertFalse(any(s['data-module'] == '1' for s in self.deck.slides))
        self.assertEqual(len(self.deck.slides), 48)
        self.assertIn('01 / 48', self.text)
        for position, module in zip(self.deck.links, ['2', '3', '4', '5', '6']):
            self.assertEqual(self.deck.slides[position]['data-module'], module)
            self.assertNotEqual(self.deck.slides[position - 1]['data-module'], module)

    def test_original_cutoff_and_added_slides(self):
        self.assertEqual([int(s['data-index']) for s in self.deck.slides],
                         [0] + list(range(9, 56)))
        self.assertIn('¿Qué pasa si usamos una red densa?', self.text)
        self.assertIn('Una capa densa combina toda la entrada', self.text)
        self.assertIn('Una imagen no es solo una lista de valores', self.text)
        self.assertNotIn('Patrones globales frente a patrones locales', self.text)
        self.assertIn('catdog.jpg', self.text)

    def test_last_slide_compares_dense_and_convolution_on_images(self):
        last = self.text.split('data-index="54"', 1)[1]
        self.assertIn('class="spatial-comparison"', last)
        self.assertIn('id="spatial-shift"', last)
        self.assertIn('id="spatial-pause"', last)
        self.assertIn('Mismos pesos, otra posición', last)
        self.assertIn('No es una clasificación completa', last)
        self.assertEqual(last.count('data-pixel-image='), 4)
        css = (ROOT / 'docs/pregrado/dense-layer.css').read_text()
        self.assertIn('@keyframes spatial-before', css)
        self.assertIn('@keyframes spatial-after', css)
        self.assertIn('#spatial-shift:checked ~ svg', css)
        self.assertIn('#spatial-pause:checked ~ svg', css)

    def test_illustrated_pixel_responses_match_the_visible_images(self):
        last = self.text.split('data-index="54"', 1)[1]
        svg = ET.fromstring(re.search(r'<svg\b.*?</svg>', last, re.S)[0])
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        for phase_name, column in [('spatial-before', 1), ('spatial-after', 4)]:
            phase = svg.find(f"svg:g[@class='{phase_name}']", ns)
            for kind in ['dense', 'conv']:
                image = phase.find(f"svg:g[@data-pixel-image='{kind}-{column}']", ns)
                pixels = list(image)[:35]
                self.assertEqual(len(pixels), 35)
                values = [int(pixel.attrib['fill'] == '#FFFFFF') for pixel in pixels]
                self.assertEqual(sum(values), 3)
                dense = sum(values[row * 7 + 1] for row in (1, 2, 3))
                conv = [sum(values[row * 7 + c] for row in (1, 2, 3))
                        for c in range(1, 6)]
                self.assertEqual(dense, 3 if column == 1 else 0)
                self.assertEqual(conv[column - 1], 3)
                self.assertEqual(sum(conv), 3)
            labels = [''.join(t.itertext()) for t in phase.findall('svg:text', ns)]
            self.assertIn(f'Respuesta: {dense}', labels)
            self.assertIn('Respuesta: 3', labels)
            self.assertEqual(labels[2:7], [str(v) for v in conv])

    def test_filter_diagram_reuses_the_previous_kernel(self):
        self.assertIn('data-index="55"', self.text)
        self.assertIn('Un filtro es una pequeña matriz de pesos', self.text)
        self.assertIn('convolution-filter.svg', self.text)
        svg = ET.parse(ROOT / 'docs/pregrado/convolution-filter.svg').getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        def values(group):
            return [int(t.text) for t in svg.findall(f".//svg:g[@id='{group}']/svg:text", ns)]
        patch = values('filter-patch')
        kernel = values('filter-weights')
        output = values('filter-map')
        self.assertEqual(patch, [0, 1, 0] * 3)
        self.assertEqual(kernel, [0, 1, 0] * 3)
        self.assertEqual(sum(p * w for p, w in zip(patch, kernel)), 3)
        self.assertEqual(output, [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0])

    def test_accessible_local_animation_and_spatial_explanation(self):
        self.assertIn('id="dense-pause"', self.text)
        self.assertIn('19 267 712', self.text)
        self.assertIn('no borra los valores', self.text)
        css = (ROOT / 'docs/pregrado/dense-layer.css').read_text()
        self.assertIn('prefers-reduced-motion: reduce', css)
        self.assertIn('#dense-pause:checked ~ svg', css)
        self.assertIn('.slide.is-active', css)
        self.assertIn('@keyframes', css)


if __name__ == '__main__':
    unittest.main()
