import torch
from torch import Generator
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_vocab(words):
    chars = sorted(list(set(''.join(words))))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    char_to_ix['.'] = 0
    return char_to_ix, {i: ch for ch, i in char_to_ix.items()}


def build_dataset(words: list, stoi: dict,
                  block_size) -> tuple[torch.Tensor, torch.Tensor]:
    _xs, _ys = [], []

    """
    Finds all bigrams (with context size) in the words and converts them
    into an input tensor and target tensor using indices (stoi dict).
    The shape of the input tensor is (x, block_size)
    The shape of the target tensor is (x,) where x is related to the number
    of characters in the words. (x = length of all words together + how many words)
    """

    for w in words:
        # initialize the context with dots
        # padding the beginning of the sentence with dots
        context = [0] * block_size
        for ch in w + '.':  # padding the end of the sentence with a dot
            idx = stoi[ch]
            _xs.append(context)
            _ys.append(idx)
            # update the context for next iteration
            context = context[1:] + [idx]

    _xs = torch.tensor(_xs)
    _ys = torch.tensor(_ys)

    return _xs, _ys


class MLP_LM():
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 g: Generator = None):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.C = torch.randn(vocab_size, embedding_dim, generator=g)
        self.W1 = torch.randn(
            context_length * embedding_dim, hidden_dim, generator=g)
        self.b1 = torch.randn(hidden_dim, generator=g)
        self.W2 = torch.randn(hidden_dim, vocab_size, generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)
        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.params:
            p.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Takes in a tensor of input data (the dataset or a slice of it)
        Returns the log-counts of the next character for each example in the input data
        """
        emb = self.C[X]
        h = torch.tanh(
            (emb.view(-1, self.context_length * self.embedding_dim) @ self.W1) + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def get_loss(self, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, Y)

    def backward(self, loss: torch.Tensor, lr: float = 0.01) -> torch.Tensor:
        """
        Takes the loss and the learning rate and updates the parameters,
        by backpropagating the gradients.
        """
        for p in self.params:
            p.grad = None
        loss.backward()
        for p in self.params:
            p.data -= lr * p.grad

    def train(self, Xtr: torch.Tensor, Ytr: torch.Tensor, iterations: int,
              batch_size: int = 32, lr: float = 0.01,
              losses_before: list[float] = []
              ) -> list[float]:
        """
        Takes the full training data and expected outputs,
        a batch size (every how many examples to update the parameters),
        and a learning rate.

        Returns the loss statistic.
        """
        losses = losses_before
        with tqdm(total=iterations) as pbar:
            pbar.set_description('Training')
            for _ in range(iterations):
                # create a minibatch slicer
                idx = torch.randint(0, Xtr.shape[0], (batch_size,))

                logits = self.forward(Xtr[idx])

                loss = self.get_loss(logits, Ytr[idx])
                losses.append(loss.item())

                self.backward(loss, lr=lr)

                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        return losses

    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Does a forward pass on the input data and returns the loss without performing a
        training step.
        """
        logits = self.forward(X)
        loss = self.get_loss(logits, Y)
        return loss.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of input data and returns the probabilities of the next character
        for each example in the input data
        """
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        return probs

    def generate(self, n: int, itos: dict) -> str:
        """
        Generates n words using the model.
        """
        generated = []
        for _ in range(n):
            _word = []
            context = [0] * self.context_length
            while True:
                probs = self.predict(torch.tensor(context))
                idx = torch.multinomial(probs, num_samples=1).item()
                if idx == 0:
                    break
                context = context[1:] + [idx]
                _word.append(itos[idx])
            generated.append(''.join(_word))
        return generated

    def plot_vocab_embedding(self, itos: dict, dim1: int, dim2: int):
        """
        Plots the vocabulary embeddings. For each dimension
        """

        _C = self.C.detach()

        plt.figure(figsize=(15, 15))

        plt.scatter(_C[:, dim1], _C[:, dim2], s=300)
        for i, txt in itos.items():
            plt.text(_C[i, dim1].item(), _C[i, dim2].item(), txt,
                     fontsize=12, ha='center', va='center', color='white')

        plt.grid('minor')
        plt.show()
