import unittest

from scripts.export_abcm1_factors import select_split_dates


class ExportScriptTests(unittest.TestCase):
    def test_select_split_dates_honors_split_and_limit(self):
        train_dates = ["20200102", "20200103", "20200106"]
        valid_dates = ["20200107", "20200108"]

        self.assertEqual(select_split_dates(train_dates, valid_dates, "train", 2), train_dates[:2])
        self.assertEqual(select_split_dates(train_dates, valid_dates, "valid", -1), valid_dates)
        self.assertEqual(select_split_dates(train_dates, valid_dates, "all", 4), train_dates[:3] + valid_dates[:1])


if __name__ == "__main__":
    unittest.main()
