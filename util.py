def print_tokens(tokens, is_bytes=False):
    """Expects the tokens to be either ints representing bytes or str"""
    for token in tokens:
        if is_bytes and isinstance(token, int):
            print(bytes([token]).decode(errors='ignore'), end='')
        else:
            print(token, end='')
    print()  # To print the end of line character


def coalesce(value, default_value):
    """https://en.wikipedia.org/wiki/Null_coalescing_operator
    Can't use 'or' because of the RuntimeError thrown by PyTorch"""
    try:
        if value:
            return value
    except RuntimeError:  # bool(torch.tensor) throws RuntimeError for lists with len > 1
        return value
    return default_value
