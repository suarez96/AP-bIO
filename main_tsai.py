import argparse
from train import run
import train_utils
import Dataloader
import Models

parser = argparse.ArgumentParser(
    description='Script to train breathing signal predictor based on ECG input.'
)

parser.add_argument('-y', '--yaml_path', type=str, help='Filepath to YAML file with training run params', default='params.yml')
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
args = vars(parser.parse_args())

def run(args):

    args['yaml_args'] = train_utils.load_yaml(args['yaml_path'])
    train_loader, test_loader = Dataloader.build_loaders(args)
    model = Models.TSAITransformer()
    assert False
    # the fastai object that manages the training loop
    learner = Learner(train_loader, model, loss_func=MSELossFlat(), metrics=rmse, cbs=ShowGraph())
    model.train(train_loader)
    model.eval(test_loader, args['metrics'])
    model.export(format='onnx')


if __name__ == '__main__':
    
    run(args)