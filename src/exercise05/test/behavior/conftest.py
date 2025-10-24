import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        # Check if the item has a 'timeout' marker
        if not any(mark.name == "timeout" for mark in item.own_markers):
            item.add_marker(pytest.mark.timeout(10))
