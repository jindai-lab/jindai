import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import click
import requests
from PIL import Image
from tqdm import tqdm

from .common import set_up_seed
from .inference_model import InferenceModel
from .trainer import Trainer, validate_and_test


def init_logging() -> None:
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


@click.group()
def cli():
    pass


@cli.command("train-model", short_help="Train model")
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=4, type=int)
@click.option("--num_epoch", help="number of epoch", default=32, type=int)
@click.option("--init_lr", help="initial learning rate", default=0.0001, type=float)
@click.option("--drop_out", help="drop out", default=0.5, type=float)
@click.option("--optimizer_type", help="optimizer type", default="adam", type=str)
@click.option("--seed", help="random seed", default=42, type=int)
@click.option("--dataset_dir", help="dataset dir", default='', type=str)
def train_model(
    batch_size: int,
    num_workers: int,
    num_epoch: int,
    init_lr: float,
    drop_out: float,
    optimizer_type: str,
    seed: int,
    dataset_dir: str
):
    init_logging()
    click.echo("Train and validate model")
    set_up_seed(seed)
    trainer = Trainer(
        experiment_dir=Path('experiments'),
        batch_size=batch_size,
        num_workers=num_workers,
        num_epoch=num_epoch,
        init_lr=init_lr,
        drop_out=drop_out,
        optimizer_type=optimizer_type,
        dataset_dir=dataset_dir
    )
    trainer.train_model()
    click.echo("Done!")


@cli.command("score-image", short_help="Get image scores")
@click.option("--model", help="path to model weight .pth file", required=True, type=Path)
@click.option("--image", help="image ", required=True, type=Path)
def get_image_score(model, image):
    model = InferenceModel(path_to_model_state=model)
    result = model.predict_from_file(image)
    click.echo(result)


@cli.command("validate-model", short_help="Validate model")
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--drop_out", help="drop out", default=0.0, type=float)
def validate_model(path_to_model_state, path_to_save_csv, path_to_images, batch_size, num_workers, drop_out):
    validate_and_test(
        path_to_model_state=path_to_model_state,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_out=drop_out,
    )
    click.echo("Done!")


@cli.command("download-dataset")
@click.option("--host", default="localhost:8370", type=str)
@click.option("--count", default=1500, type=int)
@click.option("--dataset-dir", default='', type=str)
def download_dataset(host, count, dataset_dir):

    positives = requests.post(f'http://{host}/api/search', headers={'content-type': 'application/json'}, data=json.dumps({'limit': int(count * 1.1), 'tag': '_rating>1'})).json()['results']
    negatives = requests.post(f'http://{host}/api/search', headers={'content-type': 'application/json'}, data=json.dumps({'limit': int(count * 1.1), 'tag': '_rating<=-1|((@asiantolick|@Fengsiyuan|@tbzt3623|@shuanggu888|@xihuansiwalau),_rating<1)'})).json()['results']
    # negatives = glob.glob('negatives/*.jpg'); random.shuffle(negatives); negatives = negatives[:int(limit * 1.1)]

    dirs = { t : os.path.join(dataset_dir, t) for t in ('train', 'val', 'test') }
    for tp in dirs.values():
        if not os.path.exists(tp):
            os.makedirs(tp)
    
    with tqdm(total=int(count * 1.06) * 2) as pbar:

        def download(args):
            u, dst, fn = args
            try:
                if isinstance(u, dict):
                    i = requests.get(f'http://{host}/block/' + u['images'][0]['_id'] + '.jpg?w=480').content
                    i = Image.open(BytesIO(i))
                else:
                    i = Image.open(u)
                i.save(os.path.join(dst, fn))
                pbar.update(1)
            except (OSError, TimeoutError) as ex:
                print(ex)

        def samples(l, label):
            for i, a in enumerate(l):
                f = 'train'
                if i >= int(count * 0.9): f = 'test'
                if i >= count: f = 'val'
                if i >= int(count * 1.06): continue
                yield a, f, f'{label}-{i:04}.jpg'

        with ThreadPoolExecutor(max_workers=10) as te:
            for r in te.map(download, samples(positives, 1)): pass
            for r in te.map(download, samples(negatives, 0)): pass
        

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cli()
