import numpy as np

from games.base import Card

def rank2int(rank):
    return Card.ranks.index(rank) + 2

def int2rank(num):
    return Card.ranks[num - 2]

def one_bit_set(size, index):
    vector = np.zeros(size, dtype=np.uint8)
    vector[index] = 1

    return vector

def int_to_binary_array(integer, num_bits=8):
    binary_string = bin(integer)[2:].zfill(num_bits)
    binary_array = np.array([int(bit) for bit in binary_string])
    return binary_array