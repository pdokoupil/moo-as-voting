class top_k_filter_function:
    def __init__(self, k):
        assert k >= 1, f"Value of k must be >= 0 but it is: {k}"
        self.k = k

    # Each element of the item_supports_pairs is a tuple (item, supports_value)
    def __call__(self, item_supports_pairs):
        return {item: support for item, support in sorted(item_supports_pairs, key=lambda key: key[1], reverse=True)[:self.k]}