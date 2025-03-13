import torch
import pytest
import random

from Utility.Extensions import TensorQueue

@pytest.mark.local
def test_circular_naive():
    buf = TensorQueue(torch.Size((3, 1)), grow_dim=0, device=torch.device('cpu'), dtype=torch.float32)
    x   = torch.tensor([[1.]])
    buf.push(x)
    assert (buf.tensor == torch.tensor([[1.]])).all()


@pytest.mark.local
def test_circular_random():
    buf   = TensorQueue(torch.Size((1, 10)), grow_dim=1, device=torch.device('cpu'), dtype=torch.long)
    start        = 0
    length_range = 8

    for _ in range(50):
        x_length = random.randint(0, length_range)
        x = torch.arange(start, (start := start + x_length), step=1).unsqueeze(0)
        buf.push(x)
        content = buf.tensor
        assert ((content[1:] - content[:-1]) == 1).all(), "Unexpected result."

@pytest.mark.local
def test_circular_scalar():
    buf   = TensorQueue(torch.Size((10,)), grow_dim=0, device=torch.device('cpu'), dtype=torch.long)
    for value in range(50):
        buf.push_scalar(value)
        content = buf.tensor
        assert ((content[1:] - content[:-1]) == 1).all(), "Unexpected result."


if __name__ == "__main__":
    from tqdm import tqdm
    
    device = 'cpu'
    start        = 0
    length_range_from = 1
    length_range_to   = 2
    max_size          = 1000
    read_every        = 200

    # Tensor Test (push_scalar)
    buf   = TensorQueue(torch.Size((max_size,)), grow_dim=0, device=torch.device(device), dtype=torch.long)
    for _ in tqdm(range(100000), desc="CircularTensor Solution"):
        x_length = 1
        for i in range(200):
            buf.push_scalar(i)
        content = buf.tensor

    # Tensor Test (batch push)
    buf   = TensorQueue(torch.Size((max_size,)), grow_dim=0, device=torch.device(device), dtype=torch.long)
    for _ in tqdm(range(100000), desc="CircularTensor Solution"):
        x_length = 1
        buf.push(torch.arange(1, 200, dtype=torch.long))
        content = buf.tensor

    # Deque Test
    from collections import deque
    dq    = deque([], maxlen=max_size)
    for _ in tqdm(range(100000), desc="Deque Solution"):
        x_length = 1
        for i in range(200):
            dq.append(start)
        content = torch.tensor(list(dq))
