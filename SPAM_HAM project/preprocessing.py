import re

def to_alpha(text: str) -> str:
    """
    Convert text to lowercase and remove non-alphabetic characters.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    return re.sub(r'[^a-z ]', '', text.lower())
