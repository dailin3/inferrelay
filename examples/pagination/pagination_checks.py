import json
import unittest

from pagination import paginate


class PaginationChecks(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 1, 2), [1, 2])

    def test_middle_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 2, 2), [3, 4])

    def test_partial_last_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 3, 2), [5])

    def test_page_past_end(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 4, 2), [])

    def test_empty_items(self):
        self.assertEqual(paginate([], 1, 2), [])

    def test_invalid_page(self):
        for page in (0, -1):
            with self.subTest(page=page), self.assertRaises(ValueError):
                paginate([1, 2, 3], page, 2)

    def test_invalid_page_size(self):
        for page_size in (0, -1):
            with self.subTest(page_size=page_size), self.assertRaises(ValueError):
                paginate([1, 2, 3], 1, page_size)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PaginationChecks)
    result = suite.run(unittest.TestResult())
    failed_ids = {
        test.id().split(" ", 1)[0].rsplit(".", 1)[-1]
        for test, _ in result.failures + result.errors
    }
    print(json.dumps(sorted(failed_ids)))
