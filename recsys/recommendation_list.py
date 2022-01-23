import copy

# Class representing recommendation list, a.k.a ranking -- ordered, with (possibly implicit) priorities
class recommendation_list:

    # K is the expected full length of the list
    # Items are the items currently present in the recommender list (could be less than k)
    def __init__(self, k, items = []):
        assert len(items) <= k, f"The nummber of items in the recommendation list {len(items)} cannot exceed {k}"
        self.items = items[:]
        self.k = k

    def append_item(self, item):
        if self.is_full():
            return False
        self.items.append(item)
        return True

    # Returns copy of this recommendation list with an extra item added at the end of it
    def with_extra_item(self, item):
        assert len(self) < self.k, "There must be a space for an extra item"
        new_rec_list = copy.deepcopy(self)
        new_rec_list.append_item(item)
        return new_rec_list

    def with_replaced_item(self, index, item):
        assert index >= 0 and index < len(self), "The index to be replaced must be within bounds"
        new_rec_list = copy.deepcopy(self)
        new_rec_list[index] = item
        return new_rec_list

    def is_full(self):
        return len(self.items) == self.k

    # Implement random (list-like) access to the recommendation list
    def __delitem__(self, key):
        del self.items[key]

    def __getitem__(self, key):
        return self.items[key]
    
    def __setitem__(self, key, value):
        self.items[key] = value

    # Length of the recommendation list
    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"recommendation_list(k={self.k}, items={self.items})"
    
    def __str__(self):
        return f"Recommendation list with length: {len(self.items)}, k: {self.k}, items: {self.items}"

    def __eq__(self, obj):
        return isinstance(obj, recommendation_list) and self.items == obj.items and self.k == obj.k