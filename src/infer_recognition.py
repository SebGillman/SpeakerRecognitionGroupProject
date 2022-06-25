import argparse
import functools
import os
import shutil

import numpy as np
import torch
import time
from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from data_utils.reader import load_audio, CustomDataset
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             'model use under folder')
add_arg('threshold',        float,   0.6,                     'threshold')
add_arg('audio_db',         str,    'audio_db',               'path to audio files')
add_arg('feature_method',   str,    'melspectrogram',         'method to extract features', choices=['melspectrogram', 'spectrogram'])
add_arg('resume',           str,    'models/',                'path of models')
args = parser.parse_args()
print_arguments(args)

dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# obtain model
if args.use_model == 'ecapa_tdnn':
    ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
    model = SpeakerIdetification(backbone=ecapa_tdnn)
else:
    raise Exception(f'{args.use_model} model does not exists！')
#This part is modified to just use the cpu version of pytorch
#The below is the original code
#device = torch.device("cuda")
#model.to(device)
# load model
model_path = os.path.join(args.resume, args.use_model, 'model.pth')
model_dict = model.state_dict()
#The map_location decides the device use
param_state_dict = torch.load(model_path,map_location=torch.device('cpu'))
for name, weight in model_dict.items():
    if name in param_state_dict.keys():
        if list(weight.shape) != list(param_state_dict[name].shape):
            param_state_dict.pop(name, None)
model.load_state_dict(param_state_dict, strict=False)
print(f" loading model coefficients and opt method successfully：{model_path}")
model.eval()

person_feature = []
person_name = []


# start regonition
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method)
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32)
    # predict
    feature = model.backbone(data)
    return feature.data.cpu().numpy()


# load audio files
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        print(f"Loaded {name} audio.")


# voiceprint regonition
def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)[0]
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


# register
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    record_audio = RecordAudio()

    print('\n \n \n')

    try:
        while True:
            print('\n------------------------------------------------------------------')
            select_fun = int(input("Please type in number to choose function:\n type in 0 to register new member,\n type in 1 to do voice recognition,\n type in 2 to do continuous recognition.\n"))
            if select_fun == 0:
                audio_path = record_audio.record()
                name = input("Please type in your name as new member: ")
                if name == '': continue
                register(audio_path, name)
            elif select_fun == 1:
                audio_path = record_audio.record()
                time1 = time.time()
                name, p = recognition(audio_path)
                time2 = time.time()
                print('Total Classification time = {} seconds'.format(np.round(time2-time1, 3)))
                if p > args.threshold:
                    print("The one currently speaking is %s with a similarity of %f" % (name, p))
                else:
                    print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment")
            elif select_fun == 2:
                print("\nRecording has started, press Ctrl+C to quit")
                print("[RECORDER] Listening ...... \n")
                keypress=False
                try:
                    while True:
                        audio_path = record_audio.recordconst()
                        time1 = time.time()
                        name, p = recognition(audio_path)
                        time2 = time.time()
                        print('Classification time = {} seconds'.format(np.round(time2-time1, 3)))
                        if p > args.threshold:
                            print("The one currently speaking is %s with a similarity of %f" % (name, p))
                        else:
                            print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment \n")
                except KeyboardInterrupt:
                    pass

            else:
                print('Please type either 0, 1 or 2 \n')
                
    except KeyboardInterrupt:
        pass

