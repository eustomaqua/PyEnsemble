# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from pyensemble.utils_const import check_zero, check_equal


class TestUtilConst(unittest.TestCase):

    def test_check_zero(self):
        tem = check_zero(0)
        self.assertEqual(tem != 0., True)
        tem = check_zero(0.0)
        self.assertEqual(tem != 0., True)
        tem = check_zero(1e-23)
        self.assertEqual(tem != 0., True)

    def test_check_equal(self):
        tem = check_equal(1e-12, 1e-11)
        self.assertEqual(tem, True)
        tem = check_equal(1e-6, 3e-5)
        self.assertEqual(tem, False)



if __name__ == '__main__':
    unittest.main()
