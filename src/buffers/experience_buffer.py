#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, namedtuple
import random


class ExperienceBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, max_buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=max_buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            typename="Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack(
            [e.next_state for e in experiences if e is not None])
        dones = np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
