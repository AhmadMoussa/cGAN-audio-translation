from CombinedModel import CombinedModel
from Trainer import Trainer
from utils import generate_index, print_index
from FastYielder import Yielder

#generate_index("high_pass_dataset/train", "high_pass_train_index.npy")
#generate_index("high_pass_dataset/eval", "high_pass_eval_index.npy")
#generate_index("low_pass_dataset/tiny_eval", "low_pass_tiny_eval_index.npy")
model = CombinedModel()
trainer = Trainer(model, 1000)
trainer.train_model()
