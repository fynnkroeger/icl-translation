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
