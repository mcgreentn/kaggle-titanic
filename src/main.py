import yaml
from models.simple import Simple
from datamodules.titanic import Titanic
from pytorch_lightning import Trainer
def main():

    # read params
    with open("config/settings.yml", "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # setup model
    simple_params = settings.get("models", {}).get("simple")
    model = Simple(simple_params)
    dm = Titanic(settings.get("config"), settings.get("data"))
    dm.setup(stage="fit")
    trainer = Trainer()
    trainer.train(model, dm)


main()