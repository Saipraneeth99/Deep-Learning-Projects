# -*- coding: utf-8 -*-

import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


from data_preprocessing import createMiniBatch, data_pro_train

from scipy.special import expit
# from torchsummary import summary
# from torchsummaryX import summary
# from torchviz import make_dot, make_dot_from_trace


current_directory = os.getcwd()
dataTrain = data_pro_train(current_directory+'/MLDS_hw2_1_data/training_label.json',current_directory+'/MLDS_hw2_1_data/training_data/feat',3)
train_dataloader = DataLoader(dataset = dataTrain, batch_size=128, shuffle=True, num_workers=8, collate_fn=createMiniBatch)
with open('i2w.json', 'w') as fp:
    json.dump(dataTrain.ind2words, fp)
# with open('w2i.json', 'w') as fp:
#     json.dump(dataTrain.words2ind, fp)
# with open('gw.json', 'w') as fp:
#     json.dump(dataTrain.acceptedWords, fp)
# with open('wordFrequency.json', 'w') as fp:
#     json.dump(dataTrain.wordFrequency, fp)

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, 3*hidden_size)
        self.linear2 = nn.Linear(3*hidden_size, 2*hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seqLength, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seqLength, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        attentionWeights = self.to_weight(x)
        attentionWeights = attentionWeights.view(batch_size, seqLength)
        attentionWeights = F.softmax(attentionWeights, dim=1)
        context = torch.bmm(attentionWeights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context
    
class encoderRNN(nn.Module):
    def __init__(self):
        super(encoderRNN, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(512, 512, batch_first=True, bidirectional = False)

    def forward(self, input):
        batch_size, seqLength, feat_n = input.size()
        print("batch_size, seqLength, feat_n",batch_size, seqLength, feat_n)
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seqLength, 512)

        output, hidden_state = self.gru(input)

        return output, hidden_state
    
class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(decoderRNN, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size 
        self.vocab_size = vocab_size
        self.word_dim = 1024

        # layers
        self.embedding = nn.Embedding(len(dataTrain.ind2words) + 4, 1024)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True, bidirectional = False)
        self.attention = attention(hidden_size)
        self.final_output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_final_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_final_hidden_state.size()
        print("batch_size Decoeder",batch_size)
        decoder_current_hidden_state = None if encoder_final_hidden_state is None else encoder_final_hidden_state
        decoder_currentInputWord = Variable(torch.ones(batch_size, 1)).long()
        decoder_currentInputWord = decoder_currentInputWord.cuda()
        logP_sequence = []
        predicted_sequence = []

        targets = self.embedding(targets)
        _, seqLength, _ = targets.size()

        for i in range(seqLength-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_currentInputWord).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            
            gruInput = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gruOutput, decoder_current_hidden_state = self.gru(gruInput, decoder_current_hidden_state)
            
            logProba = self.final_output_layer(gruOutput.squeeze(1))
            logP_sequence.append(logProba.unsqueeze(1))
            decoder_currentInputWord = logProba.unsqueeze(1).max(2)[1]

        logP_sequence = torch.cat(logP_sequence, dim=1)
        predicted_sequence = logP_sequence.max(2)[1]
        return logP_sequence, predicted_sequence
        
    def prediction(self, encoder_final_hidden_state, encoder_output):
        logP_sequence = []
        predicted_sequence = []
        anticipated_sequence_length = 28

        _, batch_size, _ = encoder_final_hidden_state.size()
        decoder_current_hidden_state = None if encoder_final_hidden_state is None else encoder_final_hidden_state
        decoder_currentInputWord = Variable(torch.ones(batch_size, 1)).long()  # <BOS> (batch x word index)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder_currentInputWord = decoder_currentInputWord.to(device)

        # decoder_currentInputWord = decoder_currentInputWord.cuda() if torch.cuda.is_available() else decoder_currentInputWord

        
        for i in range(anticipated_sequence_length-1):
            current_input_word = self.embedding(decoder_currentInputWord).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gruInput = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gruOutput, decoder_current_hidden_state = self.gru(gruInput, decoder_current_hidden_state)
            logProba = self.final_output_layer(gruOutput.squeeze(1))
            logP_sequence.append(logProba.unsqueeze(1))
            decoder_currentInputWord = logProba.unsqueeze(1).max(2)[1]

        logP_sequence = torch.cat(logP_sequence, dim=1)
        predicted_sequence = logP_sequence.max(2)[1]
        return logP_sequence, predicted_sequence

    def teacher_forcing_ratio(self, training_steps):
        # tf_ratio = max(0, 1 - training_steps / self.max_training_steps)
        # return tf_ratio
        return (expit(training_steps/20 +0.85)) # inverse of the logit function
    
class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, video_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_final_hidden_state = self.encoder(video_feat)
        # rather than creating the DNN again for testing, I used additional flag to determine whether it is training or prediction
        if mode == 'train':
            logP_sequence, predicted_sequence = self.decoder(encoder_final_hidden_state = encoder_final_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'prediction':
            logP_sequence, predicted_sequence = self.decoder.prediction(encoder_final_hidden_state=encoder_final_hidden_state, encoder_output=encoder_outputs)
       
        return logP_sequence, predicted_sequence


def calculateLoss(featureMat, groundTruth, lengths):
    batch_size = len(featureMat)
    predictedCaption = None
    groundTruthCaption = None
    flag = True

    lossFunction = nn.CrossEntropyLoss()
    
    for batch in range(batch_size):
        predict = featureMat[batch]
        ground_truth = groundTruth[batch]
        seqLength = lengths[batch] -1

        predict = predict[:seqLength]
        ground_truth = ground_truth[:seqLength]
        # if flag:
        #     predictedCaption = predict
        #     groundTruthCaption = ground_truth
        #     flag = False
        # else:
        #     predictedCaption = torch.cat((predictedCaption, predict), dim=0)
        #     groundTruthCaption = torch.cat((groundTruthCaption, ground_truth), dim=0)

        predictedCaption = predict if flag else torch.cat((predictedCaption, predict), dim=0)
        groundTruthCaption = ground_truth if flag else torch.cat((groundTruthCaption, ground_truth), dim=0)
        flag = False if flag else flag

    loss = lossFunction(predictedCaption, groundTruthCaption)
    avg_loss = loss/batch_size

    return loss 



def train(model, epoch, trainDataLoader):
    model.train()
    # ModelSaveLoc = 'weightsFi4_1'
    # if not os.path.exists(ModelSaveLoc):
    #     os.mkdir(ModelSaveLoc)
    model = model.cuda()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.00015)

    print('Epoch :',epoch , ' Started ')
    for batch_idx, batch in enumerate(trainDataLoader):
        video_feats, ground_truths, lengths = batch
        video_feats, ground_truths = video_feats.cuda(), ground_truths.cuda()
        video_feats, ground_truths = Variable(video_feats), Variable(ground_truths)
        
        optimizer.zero_grad()
        logP_sequence, predicted_sequence = model(video_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
            
        ground_truths = ground_truths[:, 1:]  
        loss = calculateLoss(logP_sequence, ground_truths, lengths)
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print('epoch :',epoch,' Loss ', loss)
    
#     if(loss < 1.9):
#         torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model'+str(epoch)))
    
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   
    epochs_n = 65
    ModelSaveLoc = 'weights'
    outputSize = vocabSize = len(dataTrain.ind2words) +4
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    encoder = encoderRNN()
    decoder = decoderRNN(512, outputSize, vocabSize, 1024, 0.2)
    model = MODELS(encoder=encoder, decoder=decoder)
    # summary(encoder, input_size=[(128, 80, 4096),(40, 80, 4096)])
    # print(summary(encoder,[12]
    # print("len(dataTrain.ind2words) +4" , len(dataTrain.ind2words) +4)
    for epoch in range(epochs_n):
        train(model,epoch+1,train_dataloader)
        # torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model'+str(epoch)))
        if(epoch % 16 == 0):
            torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model'+str(epoch)))
        