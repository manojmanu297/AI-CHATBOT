import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
import speech_recognition as ar
r = ar.Recognizer()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit) and (type 'voice' to use voice chat and 'stop' to stop)")
while True:
    
    sentence = input("You: ")
    if sentence == "quit":
        break
    if sentence ==  "voice":
        print(f"{bot_name}: voice chat is activated")
        while(1):
            with ar.Microphone() as source:
                
                audio = r.listen(source)
                try:
                    print("you: "+ r.recognize_google(audio))
                    sentence = tokenize(r.recognize_google(audio))
                    X = bag_of_words(sentence, all_words)
                    X = X.reshape(1, X.shape[0])
                    X = torch.from_numpy(X).to(device)

                    output = model(X)
                    _, predicted = torch.max(output, dim=1)

                    tag = tags[predicted.item()]

                    probs = torch.softmax(output, dim=1)
                    prob = probs[0][predicted.item()]
                    if prob.item() > 0.75:
                        for intent in intents['intents']:
                            if tag == intent["tag"]:
                                print(f"{bot_name}: {random.choice(intent['responses'])}")
                    if (r.recognize_google(audio) != "stop" and  prob.item() <= 0.75):
                        print(f"{bot_name}: I do not understand...")
                    d =0
                except ar.UnknownValueError :
                    print(f"{bot_name}: google speech recognition could not understand audio")
                    d = 1
                if(d!= 1 and r.recognize_google(audio) == "stop" ):
                    print(f"{bot_name}: voice chat is deactivated")
                    break
            
    else:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")
