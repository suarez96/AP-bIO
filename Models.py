from tsai.all import TST, Learner
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__():
        pass

    def eval():
        raise NotImplementedError

    def train():
        raise NotImplementedError

    def export():
        # TODO fix
        model_path = os.path.join("models", "tsai", f"clf_CSMD-seq-len-{seq_len}-iters-{iters}-samples-{TRAIN_SAMPLES}-lr-{lr}-stride-{train_jump_size}-batch-size-{batch_size}.pkl")
        if os.path.exists(model_path):
            model_path.replace(".pkl", "_new.pkl")
        learn.export(model_path)


class TSAITransformer(Model):

    def __init__(self):
        """
        """
        super().__init__()
        # Use a specific architecture directly, e.g., TST for ResNet-like behavior
        model = TST(dls.vars, dls.c, seq_len=seq_len)
        return model


class BidirectionalRNN(Model):
    
    def __init__(self):
        pass
