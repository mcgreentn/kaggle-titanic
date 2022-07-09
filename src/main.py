import yaml
import csv
from models.simple import Simple
from datamodules.titanic import Titanic
from pytorch_lightning import Trainer
from math import floor

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

    # list
    predictions = [prediction.squeeze().tolist() for prediction in predictions]
    predictions = [[floor(element[0]), floor(element[1])] for element in predictions]

    
    # print(predictions)

    fields = ["PassengerId", "Survived"]
    with open("simple_predictions.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(predictions)
    # setup random forest with boosting

if __name__ == '__main__':
    main()