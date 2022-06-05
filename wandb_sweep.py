import os
from argparse import ArgumentParser

import yaml
from jinja2 import Environment, PackageLoader

import wandb


def main(args):
    if type(args) is not dict:
        args = vars(args)
    if not args['configuration_file']:
        raise FileNotFoundError
    with open(args['configuration_file'], "r") as stream:
        config = yaml.safe_load(stream)

    setup = config['setup']
    hyperparameters = config['hyperparameters']

    env = Environment(
        loader=PackageLoader("wandb_sweep")
    )
    template = env.get_template("sweep_python.py.jinja2")

    for dataset in setup['datasets']:
        project_name = setup['project_format_string'].format(dataset=dataset)
        if not os.path.exists(os.path.join(setup['output_dir'], project_name)):
            os.mkdir(os.path.join(setup['output_dir'], project_name))
        for model in setup['models']:
            for checkpoint in setup['checkpoints']:
                sweep_name = f"{model}{checkpoint['dataset_trained_on']}"
                save_folder = os.path.join(setup['output_dir'], project_name, model)
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                with open(os.path.join(save_folder, f"{sweep_name}.py"), "w+") as f:
                    f.write(
                        template.render(dataset=dataset, model=model, checkpoint=checkpoint, project_name=project_name))

                hyperparameters.update({"name": sweep_name})
                hyperparameters.update({"program": os.path.join(save_folder, f"{sweep_name}.py")})
                sweep_id = wandb.sweep(hyperparameters, project=project_name)

                with open(os.path.join(setup['output_dir'], project_name, "sweep_agent_commands.txt"), "a+") as f:
                    f.write(f"wandb agent {setup['username']}/{project_name}/{sweep_id}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('configuration_file')

    _args = parser.parse_args()
    main(_args)
