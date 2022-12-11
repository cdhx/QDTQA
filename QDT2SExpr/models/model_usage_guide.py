# Usage guide on model and relevant datasets
import sys
sys.path.append('../..')
from src.data.inputDataset.serializationDataset import SerializationDataset
from encoder.BertSerializationEncoder import BertSerializationEncoder

import torch

def use_serialization(no_cuda=False):
    """
    对 QDT 进行序列化编码，并用 BERT 进行 embedding
    """
    # Dataset. Using default parameters
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    serialization_dataset = SerializationDataset(no_cuda=no_cuda, batch_size=32)
    train_iter, validation_iter, test_iter = serialization_dataset.get_iterators()
    serialization_vocab = serialization_dataset.get_serialization_vocab()

    # Model, using default parameters
    model = BertSerializationEncoder(verbose=True)

    # Sample code for embedding
    model.to(device)
    model.train()

    for batch in train_iter:
        embedding = model(batch.serialization)
        print("id: {}".format(batch.idx))


if __name__ == '__main__':
    use_serialization()