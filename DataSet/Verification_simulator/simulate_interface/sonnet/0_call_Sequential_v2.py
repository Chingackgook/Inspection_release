from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.sonnet import *
exe = Executor('sonnet', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/sonnet/examples/simple_mnist.py'
from typing import Dict
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds
'Trivial convnet learning MNIST.'

def mnist(split: str, batch_size: int) -> tf.data.Dataset:
    """Returns a tf.data.Dataset with MNIST image/label pairs."""

    def preprocess_dataset(images, labels):
        images = (tf.cast(images, tf.float32) / 255.0 - 0.5) * 2.0
        return (images, labels)
    dataset = tfds.load(name='mnist', split=split, shuffle_files=split == 'train', as_supervised=True)
    dataset = dataset.map(preprocess_dataset)
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def train_step(model: snt.Module, optimizer: snt.Optimizer, images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Runs a single training step of the model on the given input."""
    with tf.GradientTape() as tape:
        logits = exe.run('call', inputs=images)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply(gradients, variables)
    return loss

@tf.function
def train_epoch(model: snt.Module, optimizer: snt.Optimizer, dataset: tf.data.Dataset) -> tf.Tensor:
    """Runs a training epoch over the dataset."""
    loss = 0.0
    for (images, labels) in dataset:
        loss = train_step(model, optimizer, images, labels)
    return loss

@tf.function
def test_accuracy(model: snt.Module, dataset: tf.data.Dataset) -> Dict[str, tf.Tensor]:
    """Computes accuracy on the test set."""
    (correct, total) = (0, 0)
    for (images, labels) in dataset:
        preds = tf.argmax(model(images), axis=1)
        correct += tf.math.count_nonzero(tf.equal(preds, labels), dtype=tf.int32)
        total += tf.shape(labels)[0]
    accuracy = correct / tf.cast(total, tf.int32) * 100.0
    return {'accuracy': accuracy, 'incorrect': total - correct}
model = exe.create_interface_objects(interface_class_name='Sequential', layers=[snt.Conv2D(32, 3, 1), tf.nn.relu, snt.Conv2D(32, 3, 1), tf.nn.relu, snt.Flatten(), snt.Linear(10)])
optimizer = snt.optimizers.SGD(0.1)
train_data = mnist('train', batch_size=128)
test_data = mnist('test', batch_size=1000)
for epoch in range(5):
    train_loss = train_epoch(model, optimizer, train_data)
    test_metrics = test_accuracy(model, test_data)
    print('[Epoch %d] train loss: %.05f, test acc: %.02f%% (%d wrong)' % (epoch, train_loss, test_metrics['accuracy'], test_metrics['incorrect']))