import argparse
import os
import sys

# TF >= 2.16 ships Keras 3, which removed legacy rnn_cell.LSTMCell used by the
# BLSTM. Enable the Keras-2 compat shim before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# This project uses paths relative to the src dir (../model, ../data, ./logs).
# chdir here so `python .../main.py` works no matter where it is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Allow custom checkpoint directory via environment variable
# This enables separate checkpoints for base model vs fine-tuning
# Usage: set MODEL_DIR=../model_finetune before calling main.py
CHECKPOINT_DIR = os.environ.get('MODEL_DIR', '../model')
if not CHECKPOINT_DIR.startswith('..'):
    CHECKPOINT_DIR = f'../{CHECKPOINT_DIR}'

import cv2
try:
    import editdistance  # only needed for --validate (CER metric); no 3.13 wheel
except ImportError:
    editdistance = None
import numpy as np
import tensorflow.compat.v1 as tf

from DataLoader import Batch, DataLoader, FilePaths
from SamplePreprocessor import preprocessor, wer
from Model import DecoderType, Model
from SpellChecker import correct_sentence

# Override checkpoint paths with custom directory if specified
FilePaths.fnCharList = f'{CHECKPOINT_DIR}/charList.txt'
FilePaths.fnWordCharList = f'{CHECKPOINT_DIR}/wordCharList.txt'
FilePaths.fnAccuracy = f'{CHECKPOINT_DIR}/accuracy.txt'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(model, loader):
    """ Train the neural network """
    # Try to load last epoch from file (for resuming training)
    epochFile = '../model/last_epoch.txt'
    try:
        with open(epochFile, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
            content = f.read().strip()
            epoch = int(content)
            print('\nResuming from epoch %d (loading from checkpoint)' % (epoch + 1))
    except Exception as e:
        epoch = 0

    bestCharErrorRate = float('inf')  # Best valdiation character error rate
    noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
    earlyStopping = 25  # Stop training after this number of epochs without improvement
    maxEpochs = 10  # Fine-tuning: 10 epochs on prescription data # Maximum number of epochs to train (hard limit) - increased for continuation
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize
    print('\n' + '='*60)
    if epoch == 0:
        print('TRAINING STARTED - Checkpoint will auto-restore if exists')
    else:
        print('TRAINING RESUMED from checkpoint')
    print('='*60 + '\n')

    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # Train
        print('Train neural network')
        loader.trainSet()
        epochLosses = []
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            epochLosses.append(loss)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        avgTrainLoss = sum(epochLosses) / len(epochLosses)
        print('--- Epoch %d Training Loss: %.4f ---' % (epoch, avgTrainLoss))

        # Validate
        charErrorRate, addressAccuracy, wordErrorRate, valLoss = validate(model, loader)
        cer_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)
        address_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
            tag='addressAccuracy', simple_value=addressAccuracy)])  # Tensorboard: Track addressAccuracy
        # Tensorboard: Add address_summary to writer
        model.writer.add_summary(address_summary, epoch)
        wer_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
            tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
        # Tensorboard: Add wer_summary to writer
        model.writer.add_summary(wer_summary, epoch)

        # Print epoch summary
        print('=' * 60)
        print('EPOCH %d SUMMARY:' % epoch)
        print('  Training Loss    : %.4f' % avgTrainLoss)
        print('  Validation Loss  : %.4f' % valLoss)
        print('  Char Error Rate  : %.2f%%' % (charErrorRate*100.0))
        print('=' * 60)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('[SAVE] Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
            # Save current epoch for resuming training
            with open('../model/last_epoch.txt', 'w') as f:
                f.write(str(epoch))
        else:
            print('[NO IMPROVE] Character error rate not improved (%d/%d)' % (noImprovementSince+1, earlyStopping))
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break

        # Stop training if max epochs reached
        if epoch >= maxEpochs:
            print('Max epochs (%d) reached. Training stopped.' % maxEpochs)
            break


def char_edit_distance(a, b):
    """ Levenshtein distance between two strings.
    Uses the fast `editdistance` C extension when available, otherwise a
    pure-Python fallback (the C extension has no prebuilt wheel for Python 3.13). """
    if editdistance is not None:
        return editdistance.eval(a, b)
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def validate(model, loader):
    """ Validate neural network """
    print('Validate neural network')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []
    valLosses = []
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        # Calculate validation loss for this batch
        sparse = model.toSpare(batch.gtTexts)
        valLoss = model.sess.run(model.loss, {
            model.inputImgs: batch.imgs,
            model.gtTexts: sparse,
            model.seqLen: [Model.maxTextLen] * len(batch.imgs)
        })
        valLosses.append(valLoss)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = char_edit_distance(recognized[i], batch.gtTexts[i])
            ## editdistance
            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER)/len(totalCER)
    addressAccuracy = numWordOK / numWordTotal
    wordErrorRate = sum(totalWER)/len(totalWER)
    avgValLoss = sum(valLosses) / len(valLosses) if valLosses else 0
    print('Validation Loss: %.4f | Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
          (avgValLoss, charErrorRate*100.0, addressAccuracy*100.0, wordErrorRate*100.0))
    return charErrorRate, addressAccuracy, wordErrorRate, avgValLoss


def load_different_image():
    imgs = []
    for i in range(1, Model.batchSize):
       imgs.append(preprocessor(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
    return imgs


def generate_random_images():
    return np.random.random((Model.batchSize, Model.imgSize[0], Model.imgSize[1]))


def infer(model, fnImg):
    """ Recognize text in image provided by file path """
    img = preprocessor(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
    if img is None:
        print("Image not found")

    imgs = load_different_image()
    imgs = [img] + imgs
    batch = Batch(None, imgs)
    recognized = model.inferBatch(batch)  # recognize text

    print("Without Correction", recognized[0])
    print("With Correction", correct_sentence(recognized[0]))
    return recognized[0]


def main():
    """ Main function """
    # Opptional command line args
    parser = argparse.ArgumentParser()
    print(parser)
    parser.add_argument(
        "--train", help="train the neural network", action="store_true")
    parser.add_argument(
        "--validate", help="validate the neural network", action="store_true")
    parser.add_argument(
        "--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # Train or validate on Cinnamon dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize,
                            Model.imgSize, Model.maxTextLen, load_aug=True)

        # Execute training or validation
        if args.train:
            # persist the char list so the trained model can be used for inference later
            open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
            # Auto-restore existing checkpoint if available, else init fresh weights
            model = Model(loader.charList, decoderType, mustRestore=False)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=False)
            validate(model, loader)

    # Infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(),
                      decoderType, mustRestore=False)
        infer(model, FilePaths.fnInfer)


def infer_by_web(path, option):
    decoderType = DecoderType.BestPath
    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(),
                  decoderType, mustRestore=False)
    recognized = infer(model, path)

    return recognized


if __name__ == '__main__':
    main()
