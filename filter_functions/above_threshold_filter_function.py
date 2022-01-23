class above_threshold_filter_function:
    def __init__(self, threshold):
        self.threshold = threshold

    # Each element of the item_supports_pairs is a tuple (item, supports_value)
    def __call__(self, item_supports_pairs):
        return {item: support for item, support in filter(lambda key: key[1] > self.threshold, item_supports_pairs)}