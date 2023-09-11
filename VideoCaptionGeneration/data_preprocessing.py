# -*- coding: utf-8 -*-

import os
import torch
import json
import re
import numpy as np
import gc
gc.collect()

current_directory = os.getcwd()



class data_pro_train():
    def __init__(self,JSON_path,featurePath,word_threshold=3):
        self.JSON_file = None
        self.JSON_path = JSON_path
        self.featurePath = featurePath
        self.word_threshold = word_threshold
        
        self.acceptedWords = []
        self.wordFrequency = {}
        self.ind2words = {}
        self.words2ind = {}
        self.featureMap = {}
        self.annotatedCaptions = []

        #Functions
        self.loadJSON()
        self.getWordCountMap()
        self.getWordEncoder()
        self.wordAnnotation()
        self.loadFeatureDict()
        


    def loadJSON(self):
        with open(self.JSON_path, 'r') as f:
            file = json.load(f)
        self.JSON_file = file
        
    def getWordCountMap(self):
        count={}
        for line in self.JSON_file:
            for jsentence in line['caption']:
                sentence = re.sub('[.!,;?-]]', ' ', jsentence).split()
                for word in sentence:
                    word = word.replace('.', '') if '.' in word else word
                    word = word.lower()
                    word = re.sub(r"[^a-zA-Z0-9]+", '', word)
                    if word in count:
                        count[word] += 1
                    else:
                        count[word] = 1
                    
        self.wordFrequency = count
        
    def getWordEncoder(self):
        #This function performs encoding of all generated words. It filters the accepted words based on a minimum length threshold 
        #It generates dictionary files mapping the words to their respective indices and vice versa.

        # acceptedWords = [k for k, v in self.wordFrequency.items() if v > self.word_threshold]
        specalTokens = [('<PAD>', 0), ('<BOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        acceptedWords = list(filter(lambda x: self.wordFrequency[x] > self.word_threshold, self.wordFrequency.keys()))

        
        index2Word = {index + len(specalTokens): word for index, word in enumerate(acceptedWords)}
        word2Index = {word: index + len(specalTokens) for index, word in enumerate(acceptedWords)}

        for token, index in specalTokens:
            index2Word[index] = token
            word2Index[token] = index
    
        self.ind2words = index2Word
        self.words2ind = word2Index
        self.acceptedWords = acceptedWords
        
        
    
    def wordAnnotation(self):
        mergedData = []

        for directory in self.JSON_file:
            captions = directory.get('caption', [])
            for sentence in captions:
                newSentence = self.indexifyWords(sentence.lower())
                mergedData.append((directory['id'], newSentence))

        self.annotatedCaptions = mergedData



    def indexifyWords(self,sentence):
    # sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = re.findall(r'\b\w+\b', sentence)
        sentence = ['<BOS>'] + ['<UNK>' if (self.wordFrequency.get(word, 0) <= self.word_threshold) else word for word in sentence] + ['<EOS>']
        # sentence = ['<BOS>'] + [w if (self.wordFrequency.get(w, 0) > self.word_threshold) \
        #                             else '<UNK>' for w in sentence] + ['<EOS>']

        sentence = [self.words2ind[word] for word in sentence]
        return sentence
        
    def loadFeatureDict(self):
        featureDictionary = {}
        for file in os.listdir(self.featurePath):
            if file.endswith('.npy'):
                key = file[:-4]  # remove the '.npy' extension
                value = np.load(os.path.join(self.featurePath, file))
                featureDictionary[key] = value
        self.featureMap = featureDictionary

                      
    def __len__(self):
        return len(self.annotatedCaptions)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        video_file_name, sentence = self.annotatedCaptions[idx]
        data = torch.Tensor(self.featureMap[video_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)
    


class data_pro_test():
    def __init__(self,featurePath):
        self.featurePath = featurePath
        self.test_featureMap = []
        
        self.testloadFeatureDict()
        
    def testloadFeatureDict(self):

        for file in os.listdir(self.featurePath):
            if file.endswith('.npy'):
                key = file[:-4]  # remove the '.npy' extension
                value = np.load(os.path.join(self.featurePath, file))

            self.test_featureMap.append([key, value])

    def __len__(self):
        return len(self.test_featureMap)
    def __getitem__(self, idx):
        return self.test_featureMap[idx]
        
     
def createMiniBatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    video_data, captions = zip(*data) 
    video_data = torch.stack(video_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return video_data, targets, lengths




# train_JSON_path = current_directory+'/MLDS_hw2_1_data/training_label.json'
# train_featurePath = current_directory+'/MLDS_hw2_1_data/training_data/feat'
# test_JSON_path = current_directory+'/MLDS_hw2_1_data/testing_label.json'
# test_featurePath = current_directory+'/MLDS_hw2_1_data/testing_data/feat'
#train = data_pro(train_JSON_path,train_featurePath)
