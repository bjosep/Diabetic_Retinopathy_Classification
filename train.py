#set arguments in a json file all

import argparse
import datetime

from model import densenet121_tf
from data_generator import create_train_valid_generator
from tensorflow_addons.metrics import CohenKappa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str,
                        help="path to training images folder")
    parser.add_argument('valid_dir', type=str,
                        help="path to the validation images folder")
    parser.add_argument('--rotation_range', type=float, default=35)
    parser.add_argument('--height_shift_range', type=float, default=0.2)
    parser.add_argument('--width_shift_range', type=float, default=0.15)
    parser.add_argument('--zoom_range', type=float, default=0.2)
    parser.add_argument('--horizontal_flip', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)

    return parser.parse_args()

def train_model(train_dir,valid_dir,rotation_range,height_shift_range,
                width_shift_range,zoom_range,horizontal_flip,train_batch_size,
                valid_batch_size, epochs, save_model=True):

    train_generator, validation_generator = create_train_valid_generator(train_dir, valid_dir,
                                 rotation_range, height_shift_range,
                                 width_shift_range, zoom_range, horizontal_flip, train_batch_size,
                                 valid_batch_size)
    model = densenet121_tf()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=[CohenKappa(num_classes=5, sparse_labels=False,
                      weightage='quadratic')])

    history = model.fit(train_generator, epochs=epochs,
                        validation_data=validation_generator, verbose=1)
    if save_model:
        today_date= str(datetime.date.today())
        model.save(f'assets/densenet121_{today_date}')



if __name__ == '__main__':
    args = get_args()
    train_model(args.train_dir, args.valid_dir, args.rotation_range, args.height_shift_range,
                args.width_shift_range, args.zoom_range, args.horizontal_flip, args.train_batch_size,
                args.valid_batch_size, args.epochs)

