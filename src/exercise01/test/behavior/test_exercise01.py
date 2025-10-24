import unittest

from exercise01.return_true import return_true


class TestReturnTrue(unittest.TestCase):
    expected_return_value = True
    
    def setUp(self):
        """Executed prior to each test to initialize required variables."""
        self.expected_return_value = True
        self.return_value = return_true()

    def test_type(self):
        self.assertTrue(
            isinstance(self.return_value, bool),
            f"Expected return value to be of type bool, got {type(self.return_value)}",
        )

    def test_value(self):
        self.assertTrue(
            self.return_value == self.expected_return_value,
            f"Expected return value to be {self.expected_return_value}, got {self.return_value}.",
        )
