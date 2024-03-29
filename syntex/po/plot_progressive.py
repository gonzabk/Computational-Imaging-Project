import tensorflow as tf
from imageio import imsave
from tensorpack import (PredictConfig,  OfflinePredictor, SmartInit)
from syntex.aparse import ArgParser
from progressive_model import ProgressiveSynTex, get_data
import os
import numpy as np

STEPS_PER_EPOCH = 200
MAX_EPOCH = 4
BATCH = 1
IMAGE_SIZE = 224

ps = ProgressiveSynTex.get_parser()
ps.add("--data-folder", type=str, default="../images/single_12")
ps.add("--save-folder", type=str, default="train_log/single_model")
ps.add("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
ps.add("--max-epoch", type=int, default=MAX_EPOCH)
ps.add("--save-epoch", type=int, default=20)
ps.add("--image-steps", type=int, default=100)
args = ps.parse_args()

data_folder = args.get("data_folder")
save_folder = args.get("save_folder")
image_size = args.get("image_size", IMAGE_SIZE)
max_epoch = args.get("max_epoch",  MAX_EPOCH)
save_epoch = args.get("save_epoch", max_epoch // 10)
image_steps = args.get("image_steps", 100)

name = "epoch-100_steps-300_LR-1E-3_batch-1_beta1-0.9_beta2-0.999/"
test_ckpt = "Models/" + name + "model-30000"
test_folder = "Models"

image_size = IMAGE_SIZE
pred_config = PredictConfig(
    model=ProgressiveSynTex(args),
    session_init=SmartInit(test_ckpt),
    input_names=["pre_image_input", "image_target"],
    output_names=['stages-target/viz', 'loss_output']
)

predictor = OfflinePredictor(pred_config)
test_ds = get_data(test_folder, image_size, isTrain=False)
test_ds.reset_state()
idx = 1
losses = list()
print("------------------ predict --------------")
for pii, it in test_ds:
    output_array, loss_output = predictor(pii, it)
    if output_array.ndim == 4:

        for i in range(output_array.shape[0]):
            imsave(os.path.join(test_folder, name + "test-{}.jpg".format(idx)), output_array[i])
            idx += 1
    else:
        imsave(os.path.join(test_folder, test_folder + "/" + name + "test-{}.jpg".format(idx)), output_array)
        idx += 1
    losses.append(loss_output)
    print("loss #", idx, "=", loss_output)
print("Test and save", idx, "images to", test_folder, "avg loss =", np.mean(losses))