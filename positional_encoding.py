import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding:
    def __init__(self, max_len, embedding_dim):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        # Create postition and divison terms to apply the formula
        position = np.arange(self.max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * (-np.log(10000.0) / self.embedding_dim))

        # Create pe matrix
        positional_encoding = np.zeros((self.max_len, self.embedding_dim))

        # Apply sin to even and cos to odd terms
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)

        self.positional_encoding = positional_encoding

    def get_encoding(self, seq_len):
        return self.positional_encoding[:seq_len]

def main():
    # Testing the class
    pe = PositionalEncoding(max_len=10, embedding_dim=16)
    encoding = pe.get_encoding(5)
    print("Shape:", encoding.shape)
    print(encoding)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(encoding, cmap='viridis')
    plt.colorbar()
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()

if __name__ == '__main__':
    main()