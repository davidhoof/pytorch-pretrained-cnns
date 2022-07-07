import glob
import os
from argparse import ArgumentParser

import yaml
from jinja2 import Environment, PackageLoader
from datetime import datetime

import wandb


def main(args):
    if type(args) is not dict:
        args = vars(args)
    if not args['configuration_file']:
        raise FileNotFoundError
    with open(args['configuration_file'], "r") as stream:
        config = yaml.safe_load(stream)

    wandb_config = config['wandb']
    setup = config['setup']
    hyperparameters = config['hyperparameters']

    if not os.path.exists(setup['data_dir']):
        os.mkdir(setup['data_dir'])
    if not os.path.exists(setup['output_dir']):
        os.mkdir(setup['output_dir'])

    env = Environment(
        loader=PackageLoader("wandb_sweep")
    )
    template = env.get_template("sweep_python.py.jinja2")

    for dataset in setup['datasets']:
        project_name = wandb_config['project_format_string'].format(dataset=dataset)
        if not os.path.exists(os.path.join(wandb_config['sweep_dir'], project_name)):
            os.makedirs(os.path.join(wandb_config['sweep_dir'], project_name), exist_ok=True)
        for model in setup['models']:
            for checkpoint in setup['checkpoint_inputs']:
                sweep_name = f"{model}{checkpoint['dataset_trained_on']}"
                save_folder = os.path.join(wandb_config['sweep_dir'], project_name, model)
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)

                defaults = dict(setup)
                defaults['classifier'] = model
                del defaults['models']
                defaults['dataset'] = dataset
                del defaults['datasets']                
                defaults['load_checkpoint'] = os.path.abspath(glob.glob(
                    os.path.join(
                        setup['output_dir'],
                        f"{checkpoint['dataset_trained_on']}/{model}/version_{checkpoint['version']}/checkpoints/*.ckpt")
                )[0])
                del defaults['checkpoint_inputs']
                defaults['data_dir'] = os.path.abspath(setup['data_dir'])
                defaults['output_dir'] = os.path.abspath(setup['output_dir'])
                defaults['wandb'] = project_name

                with open(os.path.join(save_folder, f"{sweep_name}.py"), "w+") as f:
                    f.write(
                        template.render(
                            sys_path=os.path.abspath(os.getcwd()),
                            type=type,
                            str=str,
                            defaults=defaults
                        )
                    )

                hyperparameters.update({"name": sweep_name})
                hyperparameters.update({"program": os.path.abspath(os.path.join(save_folder, f"{sweep_name}.py"))})
                sweep_id = wandb.sweep(hyperparameters, project=project_name)

                with open(os.path.join(wandb_config['sweep_dir'], project_name,
                                       f"sweep_agent_commands{datetime.now().strftime('%d_%m_%Y_%H')}.txt"),
                          "a+") as f:
                    f.write(f"wandb agent {wandb_config['username']}/{project_name}/{sweep_id} {model}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('configuration_file')

    _args = parser.parse_args()
    main(_args)
