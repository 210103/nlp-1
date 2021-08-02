import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import cv2
from torchvision import transforms
from utils import ImageCnn, TextCnn


class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Data():
    def __init__(self):
        pass

    def get_info(self, question_info, answer_info):
        self.image_id = question_info['image_id']
        self.question = question_info['question'].rstrip('?')
        self.question_type = answer_info['question_type']
        self.answer_type = answer_info['answer_type']
        self.answers = [i['answer'] for i in answer_info['answers']]
        self.answers_confidence = [i['answer_confidence']
                                   for i in answer_info['answers']]

    def solve_img(self, img_folder, img_h=224, img_w=224):
        transform = transforms.ToTensor()
        img_name = img_folder+'/'+str(self.image_id)+'.jpg'
        img = cv2.imread(img_name)
        img = cv2.resize(img, (img_h, img_w))
        self.image = torch.zeros(1,3,img_h,img_w)
        self.image[0] = transform(img)

    def solve_text(self, dic, embed, longest_sentence):
        question = [dic.word2idx[word] for word in self.question.split()]
        question += [dic.word2idx['<pad>']]*(longest_sentence-len(question))
        self.question = embed(torch.tensor(question))
        answer = [dic.word2idx[word] for word in self.answers]
        self.answers = embed(torch.tensor(answer))
        answer_confidnece = [dic.word2idx[word]
                             for word in self.answers_confidence]
        self.answers_confidence = embed(torch.tensor(answer_confidnece))
    
    def extract_features(self,img_model,text_model,longest_sentence,embed_dim):
        self.image_features=img_model(self.image)
        self.question_features=text_model(self.question.reshape(1,1,longest_sentence,embed_dim))
        print(self.question_features.shape)


class DataSet():
    def __init__(self, question_file, answer_file, img_folder, question_nums=10):
        self.dic = Dictionary()
        self.question_file = question_file
        self.answer_file = answer_file
        self.image_folder = img_folder
        self.question_nums = question_nums
        self.data = [Data() for _ in range(question_nums)]
        self.longest_sentence = 0

    def get_dic(self, embedding_dim=20):

        with open(self.question_file, "r") as f:
            data = json.load(f)['questions'][0:self.question_nums]
            data = [i['question'].rstrip('?')+'\n' for i in data]
            for sentence in data:
                sentence_list=sentence.split()
                if len(sentence_list) > self.longest_sentence:
                    self.longest_sentence = len(sentence_list)
                for word in sentence_list:
                    self.dic.add_word(word)

        with open(self.answer_file, "r") as f:
            data = json.load(f)["annotations"][0:self.question_nums]
            for i in data:
                answers = i['answers']
                for j in answers:
                    self.dic.add_word(j['answer'])
                    self.dic.add_word(j['answer_confidence'])

        self.dic.add_word('<pad>')
        self.embed = nn.Embedding(len(self.dic), embedding_dim)

    def get_data(self):

        fq = open(self.question_file)
        fa = open(self.answer_file)
        question_data = json.load(fq)['questions'][0:self.question_nums]
        answer_data = json.load(fa)["annotations"][0:self.question_nums]
        fq.close()
        fa.close()

        for i in range(self.question_nums):
            self.data[i].get_info(question_data[i], answer_data[i])
            self.data[i].solve_img(self.image_folder)
            self.data[i].solve_text(self.dic, self.embed,self.longest_sentence)
        
    def extract_features(self,img_model,text_model):
        for data in self.data:
            data.extract_features(img_model,text_model,self.longest_sentence,self.embed.weight.size(1))
        

    def __len__(self):
        return(self.question_nums)


if __name__ == '__main__':
    dataset = DataSet("./v2_OpenEnded_mscoco_train2014_questions.json",
                      "./v2_mscoco_train2014_annotations.json", "./imgs")
    dataset.get_dic()
    dataset.get_data()
    dataset.extract_features(ImageCnn(),TextCnn())
