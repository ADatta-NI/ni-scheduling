import random
from constants import (
    SCHEDULING_ENV_RANDOM_SEED
)

class Bag:
    """A data structure to store unique items and provides quicker insertion, removal and random selection APIs. 
    
    On average,
        - O(1) insertion
        - O(1) removal
        - O(1) random element selection
    """

    def __init__(self) -> None:
        self.itemList = []
        self.itemIndexMap = {}
        random.seed(SCHEDULING_ENV_RANDOM_SEED)

    def __iter__(self):
        return iter(self.itemList)

    def insert(self, item) -> None:
        """Inserts an item into the bag
        """
        if item in self.itemIndexMap:
            raise ValueError("Bag already contains the item which is requested to insert.")
        
        self.itemList.append(item)
        self.itemIndexMap[item] = len(self.itemList) - 1

    def remove(self, item) -> None:
        """Removes the specified item from the bag
        """
        if item not in self.itemIndexMap:
            raise ValueError("Bag doesn't contain the item which is requested to remove.")
        
        itemIdx = self.itemIndexMap[item]
        if len(self.itemList) != 0:
            lastItemIdx = len(self.itemList) - 1
            self.itemIndexMap[self.itemList[lastItemIdx]] = itemIdx
            self.itemList[itemIdx], self.itemList[lastItemIdx] = self.itemList[lastItemIdx], self.itemList[itemIdx]

        self.itemList.pop()
        self.itemIndexMap.pop(item)

    def sample(self) -> any:
        """Randomly samples and returns one item from the bag if non-empty, else returns None
        """
        if len(self.itemList) == 0:
            return None
        
        return random.choice(self.itemList)

    def size(self) -> int:
        """Returns the size of bag
        """
        return len(self.itemList)

    def show(self) -> None:
        """Prints the contents of the bag to console
        """
        for item in self.itemList:
            print(item)
