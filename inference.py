from sst.tokenizer import SSTTokenizer
from utils import load_saved_model
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-r", "--root", is_flag=True, help='SST root or all')
@click.option("-b", "--binary", is_flag=True, help='SST binary or fine')
@click.option("-f", "--filename", help="saved keras model file paths")
@click.option("-t", "--text", help="input text for inference")
def inference_runner(root, binary, filename, text):
    """
     SST Model Inference
    """
    inference(root, binary, filename, text)

def inference(root, binary, filename, text):
    pass


if __name__ == "__main__":
    inference_runner()