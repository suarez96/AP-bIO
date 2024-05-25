import argparse
from train import run

parser = argparse.ArgumentParser(
    description='Script to train breathing signal predictor based on ECG input.'
)

parser.add_argument('-y', '--yaml_path', type=str, help='Filepath to YAML file with training run params', default='params.yml')
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
args = parser.parse_args()

if __name__ == '__main__':
    
    run(args)

    # X_train, X_test, y_train, y_test, _, test_idxs = make_train_test_sets(
    #     dataset=marsh_dataset,
    #     ECG_transforms=ECG_transforms,
    #     IP_transforms=IP_transforms,
    #     test_size=5, 
    #     random_seed=42,
    #     randomize=args.randomize,
    #     train_idxs=args.train_indices,
    # )

    # train_params = {
    #     "n_estimators": 2000,

    # }
    # model = train(
    #     X_train,
    #     y_train,
    #     sample_size=args.sample_size,
    #     hyper_params=train_params,
    #     randomize=args.randomize
    # )
    # predictions = predict(model, X_test)

    # results = evaluate(
    #     predictions,
    #     y_test,
    #     metrics={
    #         "MSE": mean_squared_error,
    #         "R-squared": r2_score
    #     },
    #     results_file="trials.csv",
    #     params={**train_params, **dict(vars(args))}
    # )

    # output_name = build_model_name(vars(args), test_idxs)

    # # save model to onnx
    # export(model, input_shape=X_train.shape[1], output_path=output_name)