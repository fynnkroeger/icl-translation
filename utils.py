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
