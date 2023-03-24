import random
class ReservoirSampler ():
    def __init__ (self, k):
        self.reservoir = list ()
        self.max_limit = k
        self.currently_filled = 0

    def __len__ (self):
        return len (self.reservoir)

    def add (self, new_item):
        if self.currently_filled < self.max_limit:
            # case when the reservoir is not filled completely.
            # in this case, we simply copy the element to the reservoir
            self.reservoir.append (new_item)
        else:
            # case when the reservoir is filled completely.
            # in this case, we decide if we want to ignore the new item or
            # replace an existing item with the new item
            j = random.randrange(self.currently_filled)
            # if the randomly picked index is smaller than the reservoir size
            # then replace the item present at the index with the new item;
            # otherwise, ignore the new item.
            if j < self.max_limit:
                self.reservoir[j] = new_item

        self.currently_filled += 1