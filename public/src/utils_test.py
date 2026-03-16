import unittest
from typing import Any, Dict, Optional

import numpy as np
import torch


def restore_attributes_from_dict(obj: Any, attributes: Optional[Dict[str, Any]]) -> None:
    """Restore class attributes for any object."""
    if not attributes:
        return
    for attr, value in attributes.items():
        setattr(obj, attr, value)


def extract_attributes_from_dict(d: dict, attr: list):
    """Extract attributes from a dictionary."""
    return {k: v for k, v in d.items() if k in attr}


def compare_dicts(obj: unittest.TestCase, dict1, dict2, rtol=1e-4, atol=1e-5, key_list=[]):
    """Recursively compare two dictionaries."""
    if not isinstance(dict1, type(dict2)):
        obj.fail(f"Type mismatch: {type(dict1)} vs {type(dict2)}")

    if isinstance(dict1, (float, int, str)):
        obj.assertAlmostEqual(dict1, dict2, f"Values differ for key '{key_list}'")
        return

    key_list_old = key_list.copy()

    for key in dict1:
        key_list = key_list_old.copy()
        key_list.append(key)
        obj.assertIn(key, dict2, f"Key '{key_list}' missing in second dictionary")
        if isinstance(dict1[key], dict):
            compare_dicts(obj, dict1[key], dict2[key], rtol=rtol, atol=atol, key_list=key_list)
        elif isinstance(dict1[key], np.ndarray):
            # TODO add precise error message (which array is incorrect?)
            np.testing.assert_allclose(dict1[key], dict2[key], rtol=rtol, atol=atol)
        elif isinstance(dict1[key], list) or isinstance(dict1[key], tuple):
            obj.assertEqual(
                len(dict1[key]), len(dict2[key]), f"List lengths differ for key '{key_list}'"
            )
            for item1, item2 in zip(dict1[key], dict2[key]):
                compare_dicts(obj, item1, item2, rtol=rtol, atol=atol, key_list=key_list)
        elif isinstance(dict1[key], float):
            obj.assertAlmostEqual(
                dict1[key], dict2[key], delta=atol, msg=f"Values differ for key '{key_list}'"
            )
        elif isinstance(dict1[key], torch.Tensor):
            obj.assertEqual(
                dict1[key].shape,
                dict2[key].shape,
                f"Shape mismatch: {dict1[key].shape} vs {dict2[key].shape}",
            )
            obj.assertTrue(
                torch.allclose(dict1[key], dict2[key], rtol=rtol, atol=atol),
                f"Tensors differ for key '{key_list}'",
            )
        else:
            obj.assertEqual(dict1[key], dict2[key], f"Values differ for key '{key_list}'")


def gen_error_msg_array(name, computed, reference, rtol, atol, max_mismatches=2, only_name=True):
    """Generate a concise error message for array comparison failures.

    Shows the first few multidimensional indices where mismatches occur.
    """
    msg = [f"{name} mismatch."]
    if only_name:
        return "\n".join(msg)
    computed = np.asarray(computed)
    reference = np.asarray(reference)
    if computed.shape != reference.shape:
        msg.append(f"Shape mismatch: computed {computed.shape}, reference {reference.shape}")
        return "\n".join(msg)
    abs_diff = np.abs(computed - reference)
    rel_diff = abs_diff / (np.abs(reference) + atol)
    # Find all indices where the difference exceeds tolerance
    mismatch_mask = abs_diff > (atol + rtol * np.abs(reference))
    mismatch_indices = np.argwhere(mismatch_mask)
    num_mismatches = mismatch_indices.shape[0]
    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)
    msg.append(f"Max absolute difference: {max_abs:.4g}")
    msg.append(f"Max relative difference: {max_rel:.4g}")
    msg.append(f"rtol={rtol}, atol={atol}")
    if num_mismatches > 0:
        msg.append(f"Number of mismatches: {num_mismatches}")
        msg.append(
            f"First {min(max_mismatches, num_mismatches)} mismatches (index: computed vs reference):"
        )
        for idx in mismatch_indices[:max_mismatches]:
            idx_tuple = tuple(idx)
            msg.append(
                f"  {idx_tuple}: {computed[idx_tuple]!r} vs {reference[idx_tuple]!r} "
                f"(abs diff: {abs_diff[idx_tuple]:.4g}, rel diff: {rel_diff[idx_tuple]:.4g})"
            )
    else:
        msg.append("No mismatches found within given tolerances.")
    return "\n".join(msg)
