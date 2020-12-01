import fasttext
import os
from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_integer("dim", 1024,
                     "specify a integer which is dim of amino acid vector")

flags.DEFINE_integer("epoch", 100,
                     "specify a integer which is the number of training epoch")

flags.DEFINE_string("output", "./",
                    "specify an output directory")


def main(argv):
    path = "data"

    dim_size = FLAGS.dim
    epochs = FLAGS.epoch

    train_name = "train_raw.csv"

    file_name = "emb_{}.bin".format(dim_size)

    if not os.path.isfile(os.path.join(path, file_name)):
        model = fasttext.train_supervised(
            os.path.join(path, train_name),
            dim=dim_size,
            epoch=epochs,
            loss='hs'
        )
        model.save_model(os.path.join(FLAGS.output, file_name))

    print("TRAINING IS FINISHED")


if __name__ == "__main__":
    app.run(main)
