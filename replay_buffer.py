import random

class ReplayBuffer:
    """
    Constructs a buffer object that stores the past moves and samples a set of 
    subsamples. Since speed is a priority, this buffer takes the form of a 
    Ring Buffer. Samples taken are NOT GUARANTEED to be independent, and will
    often come in the same sequence they are inserted into the buffer. However,
    every time the buffer is fully traversed by either the insertion or 
    retrieval pointers, the buffer will be shuffled.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None for x in range(buffer_size)]
        self.insert_point = 0
        self.retrieve_point = 0
        self.count = 0

    def add(self, n1, n2, n3, n4, n5):
        """Add an experience to the buffer"""
        experience = (n1, n2, n3, n4, n5)
        if self.insert_point > self.buffer_size - 1:
            # SHUFFLE BUFFER
            self.insert_point = 0
            random.shuffle(self.buffer)
        if self.insert_point > self.count:
            self.count = self.insert_point
        self.buffer[self.insert_point] = experience
        # comment out the following line to get a "dummy memory"
        self.insert_point += 1

    def size(self):
        return self.count

    def sample(self, batch_size):
        """
        Samples a total of elements equal to batch_size from buffer if buffer 
        contains enough elements. Otherwise, return most recent elements.
        """

        batch = []

        # if getting a full batch from the latest retrieve point would 
        # overflow the buffer, get the most recent elements and reset the 
        # retrieve point
        if self.count - self.retrieve_point < batch_size:
            batch = self.buffer[self.count - batch_size:self.count]
            self.retrieve_point = 0
        else:
            if self.retrieve_point + batch_size > self.count:
                # SHUFFLE BUFFER
                self.retrieve_point = 0
                random.shuffle(self.buffer)
            batch = self.buffer[self.retrieve_point:self.retrieve_point + batch_size]
            self.retrieve_point += batch_size

        return batch