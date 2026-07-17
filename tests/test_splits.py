import unittest

from abcm.splits import make_time_block_folds, select_validation_fold


class SplitTests(unittest.TestCase):
    def test_time_block_folds_are_ordered_and_disjoint(self):
        dates = [f"202001{day:02d}" for day in range(1, 13)]

        folds = make_time_block_folds(dates, n_folds=3)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0].valid_dates, dates[:4])
        self.assertEqual(folds[1].valid_dates, dates[4:8])
        self.assertEqual(folds[2].valid_dates, dates[8:])
        for fold in folds:
            self.assertTrue(set(fold.train_dates).isdisjoint(fold.valid_dates))

    def test_select_validation_fold_uses_last_fold_by_default(self):
        dates = [f"202001{day:02d}" for day in range(1, 11)]

        train_dates, valid_dates = select_validation_fold(dates, n_folds=5)

        self.assertEqual(valid_dates, dates[8:])
        self.assertEqual(train_dates, dates[:8])


if __name__ == "__main__":
    unittest.main()
