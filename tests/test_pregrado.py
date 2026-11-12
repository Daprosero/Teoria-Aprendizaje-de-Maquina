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

    def slide(self, index):
        return self.text.split(f'data-index="{index}"', 1)[1].split('</article>', 1)[0]

    def test_removed_module_and_reachable_remaining_modules(self):
        self.assertNotIn('Datos: de crudos a utilizables', self.text)
        self.assertFalse(any(s['data-module'] == '1' for s in self.deck.slides))
        self.assertEqual(len(self.deck.slides), 56)
        self.assertIn('01 / 56', self.text)
        self.assertNotIn('01 / 50', self.text)
        for position, module in zip(self.deck.links, ['2', '3', '4', '5', '6']):
            self.assertEqual(self.deck.slides[position]['data-module'], module)
            self.assertNotEqual(self.deck.slides[position - 1]['data-module'], module)

    def test_original_cutoff_and_added_slides(self):
        self.assertEqual([int(s['data-index']) for s in self.deck.slides],
                         [0] + list(range(9, 64)))
        self.assertTrue(all(s['data-module'] == '6' for s in self.deck.slides[-6:]))
        self.assertIn('¿Qué pasa si usamos una red densa?', self.text)
        self.assertIn('Una capa densa combina toda la entrada', self.text)
        self.assertIn('Una imagen no es solo una lista de valores', self.text)
        self.assertNotIn('Patrones globales frente a patrones locales', self.text)
        self.assertIn('catdog.jpg', self.text)

    def test_new_slides_each_mount_one_animated_lesson(self):
        mounts = {54: 'data-spatial-lesson="comparison"',
                  55: 'data-spatial-lesson="kernel"',
                  56: 'data-spatial-lesson="filters"',
                  57: 'data-matrix-lesson'}
        for index, mount in mounts.items():
            slide = self.slide(index)
            self.assertEqual(self.text.count(mount), 1, mount)
            self.assertIn(mount, slide)
            figure = slide.split('<figure class="slide-figure', 1)[1].split('</figure>', 1)[0]
            self.assertIn(mount, figure)
            self.assertNotIn('<img', figure)
            self.assertNotIn('<svg', figure)
        self.assertNotIn('convolution-filter.svg', self.text)
        self.assertNotIn('class="spatial-comparison"', self.text)

    def test_operation_slides_each_mount_one_animated_lesson(self):
        mounts = {58: 'data-operation-lesson="stride"',
                  59: 'data-operation-lesson="padding"',
                  60: 'data-operation-lesson="maxpool"',
                  61: 'data-operation-lesson="averagepool"',
                  62: 'data-regularization-lesson="dropout"',
                  63: 'data-regularization-lesson="pipeline"'}
        for index, mount in mounts.items():
            slide = self.slide(index)
            self.assertEqual(self.text.count(mount), 1, mount)
            figure = slide.split('<figure class="slide-figure', 1)[1].split('</figure>', 1)[0]
            self.assertIn(mount, figure)
            self.assertNotIn('<img', figure)
            self.assertNotIn('<svg', figure)
            self.assertEqual(slide.count('<li>'), slide.count('</li>'))
            self.assertTrue(3 <= slide.count('<li>') <= 4, index)

    def test_operation_slides_state_dimensions_and_settings(self):
        stride = self.slide(58)
        for fragment in ['5 × 5', '3 × 3', 'paso 1', 'paso 2', '2 × 2']:
            self.assertIn(fragment, stride)
        padding = self.slide(59)
        for fragment in ['relleno 0', 'relleno 1', '5 × 5', '3 × 3', 'ceros',
                         'no agregan información']:
            self.assertIn(fragment, padding)
        for index in (60, 61):
            pool = self.slide(index)
            for fragment in ['4 × 4', '2 × 2', 'paso 2', 'sin pesos']:
                self.assertIn(fragment, pool)
        self.assertIn('max pooling', self.slide(61).lower())
        self.assertIn('3.75', self.slide(61))

    def test_dropout_and_pipeline_copy(self):
        dropout = self.slide(62)
        for fragment in ['entrenamiento', 'inferencia', 'p = 0.5', '0.5',
                         'máscara', 'no elimina neuronas', 'no achica']:
            self.assertIn(fragment, dropout.lower())
        self.assertIn('sin reescalar', dropout)
        pipeline = self.slide(63)
        for fragment in ['6 × 6', '4 × 4 × 2', 'ReLU', '2 × 2 × 2', '8', 'puntajes',
                         'ilustrativos', 'fila y una columna de ceros', 'negativos',
                         'no se aplica']:
            self.assertIn(fragment, pipeline)
        self.assertIn('6</strong> y <strong>3', pipeline)

    def test_new_slides_explain_parameters_filters_and_matrix(self):
        self.assertIn('1250', self.slide(54))
        self.assertIn('10', self.slide(54))
        self.assertIn('ilustrativos', self.slide(56))
        self.assertIn('redundan', self.slide(56))
        final = self.slide(57)
        for fragment in ['X_patches', '9 × 9', '9 × 3', 'mapas 3 × 3', 'ventanas']:
            self.assertIn(fragment, final)
        self.assertIn('cs231n.github.io/convolutional-networks', self.text)
        self.assertIn('github.com/vdumoulin/conv_arithmetic', self.text)

    def test_animation_assets_load_in_order(self):
        head = self.text.split('</head>', 1)[0]
        css = [head.index(f'href="{name}"') for name in
               ['dense-layer.css', 'animations/spatial.css', 'animations/matrix.css',
                'animations/operations.css', 'animations/regularization.css']]
        self.assertEqual(css, sorted(css))
        scripts = re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', self.text)
        local = [s for s in scripts if not s.startswith('http')]
        self.assertEqual(local, ['../assets/js/app.js', 'animations/spatial.js',
                                 'animations/matrix.js', 'animations/operations.js',
                                 'animations/regularization.js'])

    def test_local_asset_references_resolve(self):
        refs = re.findall(r'<(?:img|script|link|image)\b[^>]*?\b(?:src|href)="([^"]+)"',
                          self.text)
        local = [r for r in refs if not re.match(r'(https?:)?//|data:|#|mailto:', r)]
        self.assertTrue(local)
        for ref in local:
            self.assertTrue((DECK.parent / ref.split('#')[0]).resolve().is_file(), ref)

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
