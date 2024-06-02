import torch

def bit_cypher(b):
    N = (2 ** b) -1


    U = torch.zeros((N, b))
    V = torch.zeros((N, b))

    B = [torch.zeros(b)]
    
    for k in range(1, b+1):
        B.append([])

    I = torch.eye(b)
    
    i, j, k = 0, 0, 1

    for n in range(N):
        while torch.all(V[n] == 0):
            u = torch.abs(B[k-1][j] - I[i])

           
            if torch.sum(u) == k and not any(torch.allclose(u, vec) for vec in B[k]):
                B[k].append(u)
                V[n] = u / torch.norm(u, p=1)
                U[n] = u

            j += 1
            if j == len(B[k-1]):
                j = 0
                i += 1
                if i == b:
                    if k == 1:
                        I = torch.flip(I, dims=[0])
                    i = 0
                    B[k] = list(reversed(B[k]))
                    k += 1

    vocab_size = N
    return U, V, vocab_size


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def create_tokenizer(file_path):
    
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Customize post-processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Customize decoding
    tokenizer.decoder = decoders.ByteLevel()

    # Read the file and prepare for training
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Initialize a BPE trainer with special tokens
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    # Train the tokenizer
    tokenizer.train_from_iterator(lines, trainer=trainer)

    return tokenizer