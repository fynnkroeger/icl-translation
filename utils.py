def is_punctuation(s):
    return all(c in ".,!? ->#\n" for c in s)


def extend_left_non_alpha(tokens, index):
    stop_indices = []
    while is_punctuation(tokens[index]):
        stop_indices.append(index)
        index -= 1
    stop_indices.reverse()
    return stop_indices


def flat(a):
    out = []
    for x in a:
        out += x
    return out


def coords(from_tokens: list[int], to_tokens: list[int]):
    return [(i, j) for i in from_tokens for j in to_tokens if i < j]


def coords_multi(from_tokens: list[list[int]], to_tokens: list[list[int]]):
    out = []
    for a, b in zip(from_tokens, to_tokens):
        out += coords(a, b)
    return out


def append_pointwise(*args):
    out = [[] for _ in args[0]]
    for x in args:
        for i, a in enumerate(x):
            out[i] += a
    return out


def split_list(lst, delimiter):
    result = []
    current_sublist = []

    for item in lst:
        if item == delimiter:
            if current_sublist:
                result.append(current_sublist)
                current_sublist = []
        else:
            current_sublist.append(item)

    if current_sublist:
        result.append(current_sublist)

    return result


# by claude
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform


class SegmentedScale(ScaleBase):
    name = "segmented"

    def __init__(self, axis, **kwargs):
        # print(self, axis, kwargs)
        super().__init__(axis)
        self.breakpoint = kwargs.get("breakpoint", 1)
        self.scale_ratio = kwargs.get("scale_ratio", 10)

    def get_transform(self):
        return SegmentedTransform(self.breakpoint, self.scale_ratio)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(plt.AutoLocator())
        axis.set_major_formatter(plt.ScalarFormatter())


class SegmentedTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, breakpoint, scale_ratio):
        Transform.__init__(self)
        self.breakpoint = breakpoint
        self.scale_ratio = scale_ratio

    def transform_non_affine(self, a):
        return np.where(
            a <= self.breakpoint, a, self.breakpoint + (a - self.breakpoint) / self.scale_ratio
        )

    def inverted(self):
        return InvertedSegmentedTransform(self.breakpoint, self.scale_ratio)


class InvertedSegmentedTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, breakpoint, scale_ratio):
        Transform.__init__(self)
        self.breakpoint = breakpoint
        self.scale_ratio = scale_ratio

    def transform_non_affine(self, a):
        return np.where(
            a <= self.breakpoint, a, self.breakpoint + (a - self.breakpoint) * self.scale_ratio
        )

    def inverted(self):
        return SegmentedTransform(self.breakpoint, self.scale_ratio)


LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
    "ro": "Romanian",
    "gu": "Gujarati",
}

prompt_names = dict(
    format_single_message_arrow_title="title arrow",
    format_single_message_labeled="title label",
    format_single_message_arrow="arrow",
    format_single_message_arrow_oneline="arrow oneline",
)
print_names = dict(
    format_single_message_arrow_title="arrow title",
    format_single_message_labeled="label title",
    format_single_message_arrow="arrow",
    format_single_message_arrow_oneline="arrow oneline",
)

if __name__ == "__main__":
    from inference import *

    few = [
        dict(source="Guten Morgen!", target="Good Morning!"),
        dict(source="Fische schwimmen", target="Fish are swimming"),
    ]
    for func in [
        format_single_message_arrow_title,
        format_single_message_labeled,
        format_single_message_arrow,
        format_single_message_arrow_oneline,
    ]:
        prompt = func(few, "Ich mag Wasser.", "German", "English")
        text = prompt[-1]["content"]
        print(text)
        print()
        print()
