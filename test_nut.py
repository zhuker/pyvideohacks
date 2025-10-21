import unittest

from videohacks import _get_v_len


class NutTestCase(unittest.TestCase):
    def test_get_v_len(self):
        self.assertEqual(1, _get_v_len(0))
        self.assertEqual(1, _get_v_len(10))
        self.assertEqual(1, _get_v_len(100))
        self.assertEqual(2, _get_v_len(1000))
        self.assertEqual(2, _get_v_len(10000))
        self.assertEqual(3, _get_v_len(100000))
        self.assertEqual(3, _get_v_len(1000000))
        self.assertEqual(4, _get_v_len(10000000))
        self.assertEqual(4, _get_v_len(100000000))
        self.assertEqual(5, _get_v_len(1000000000))
        self.assertEqual(5, _get_v_len(10000000000))
        self.assertEqual(6, _get_v_len(100000000000))
        self.assertEqual(6, _get_v_len(1000000000000))
        self.assertEqual(7, _get_v_len(10000000000000))
        self.assertEqual(7, _get_v_len(100000000000000))
        self.assertEqual(8, _get_v_len(1000000000000000))


if __name__ == '__main__':
    unittest.main()
