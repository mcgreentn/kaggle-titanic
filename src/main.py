import yaml
from models.simple import Simple
from datamodules.titanic import Titanic
from pytorch_lightning import Trainer
def main():

    # read params
    with open("config/settings.yml", "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # setup simple model
    simple_params = settings.get("models", {}).get("simple")
    model = Simple(simple_params)
    dm = Titanic(settings.get("config"), settings.get("data"))
    # dm.setup(stage="fit")
    trainer = Trainer(max_epochs=simple_params.get("epochs"))
    # trainer.fit(model, dm)

    dm.setup(stage="test")
    trainer.fit(model, dm.train_dataloader())
    predictions = trainer.predict(model, dm.test_dataloader())
    print(predictions)
    # setup random forest with boosting

if __name__ == '__main__':
    main()