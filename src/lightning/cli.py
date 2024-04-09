from lightning.pytorch.cli import LightningCLI
from mnist_classification import MNISTClassificationTask, MNISTDataModule

def main():
    """
    Run 'python cli.py fit --model MNISTClassificationTask --data MNISTDataModule'
    """
    cli = LightningCLI(MNISTClassificationTask, MNISTDataModule)

if __name__ == "__main__":
    main()
