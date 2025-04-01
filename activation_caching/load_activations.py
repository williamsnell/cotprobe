import torch as t
import polars as pl
import tempfile
import zipfile
from pathlib import Path
from nnsight import LanguageModel
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union, List, Dict


def get_index(file: Path) -> int:
    return int(file.stem.split("-")[0])


def get_batch(files, batch_size):
    pass


def update_token_lengths(df: pl.DataFrame, model: LanguageModel) -> pl.DataFrame:
    shapes = [model.tokenizer(row[1], return_tensors="pt")["input_ids"].shape
              for row in df.iter_rows()]


    return df.insert_column(-1, pl.Series("shape", shapes))


def load_tensor(file, df, hidden_size):
    index = get_index(file)
    shape = t.Size([*df["shape"][index]] + [hidden_size])

    try:
        with tempfile.TemporaryDirectory() as tempdir:
            zipfile.ZipFile(file).extractall(tempdir)

            tensor = t.from_file(str(Path(tempdir) / file.stem / "data/2"), size=shape.numel(), dtype=t.bfloat16).clone()
    except Exception as e:
        print(file)
        raise e

    tensor = tensor.reshape(shape)

    return tensor


class SavedActivationDataset(Dataset):
    def __init__(self, path_to_activations: Union[str, Path], df: pl.DataFrame, positions_to_predict=None, hidden_size=8192):
        act_dir = Path(path_to_activations).expanduser()

        self.activations = {
                    i: path 
                    for i, path in enumerate(act_dir.glob("*.pt*"))
                }

        self.df = df
        self.hidden_size = hidden_size
        self.positions_to_predict = -positions_to_predict if positions_to_predict is not None else None

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, index):
        label_index = int(self.activations[index].stem.split("-")[0])
        inputs = load_tensor(self.activations[index], self.df, self.hidden_size)
        labels = t.ones(inputs.shape[:-1], dtype=t.int) * self.df["numeric label"][label_index]

        return {
                "inputs": inputs[0, self.positions_to_predict:],
                "labels": labels[0, self.positions_to_predict:],
                }


def collate_fn(batch: List[Dict[str, t.Tensor]], padding_value=-100):
    # Appropriately collate the different-length batches into one
    # tensor

    # Default is to pad on the right, which is good because
    # we don't want to propagate our loss via attention through
    # the padding tokens.)

    # Pad the inputs with 0's
    inputs = t.nn.utils.rnn.pad_sequence(
            [datapoint["inputs"] for datapoint in batch],
            padding_value=0).swapaxes(0, 1)

    # Pad the labels with -100's so we ignore them in the loss
    labels = t.nn.utils.rnn.pad_sequence(
            [datapoint["labels"] for datapoint in batch],
            padding_value=padding_value).swapaxes(0, 1)


    return {
        "inputs": inputs,
        "labels": labels,
    }



def test_ser_de():
    # Get an activation from the remote

    # Save it
    activation = None

    # Load it in

    # Verify it matches exactly


if __name__ == '__main__':
    model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    files = Path("activations").glob("*.ptrom")

    df = pl.read_ipc("responses_70b_with_sizes.arrow")

    dataset = SavedActivationDataset("activations", df)

    # load_tensor(f, df, model.config.hidden_size)


    assert list(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn))[0] == dataset[0]

