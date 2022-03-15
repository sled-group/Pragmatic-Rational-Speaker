import os
import json
import torch
import pickle
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

SIMPLICITY = 'b'


# # Read embedding
data_folder = 'input/'
bert_emb = pickle.load(open(os.path.join(data_folder, 'BERT_EMB_n_hypernym.pkl'), 'rb'))
bert_emb = torch.tensor(bert_emb)
emb_dim = 768
# with open(os.path.join(data_folder, 'wordmap.json'), 'r') as j:
# 	word_map = json.load(j)


# Read obj detection category
with open(os.path.join('input/', 'v2idx.json'), 'r') as j:
	v2idx = json.load(j)
idx2v = {v:k for k,v in v2idx.items()}



class MyDatasetPair(torch.utils.data.Dataset):
	def __init__(self, data_folder, split, maxx, bert_emb, word_map, dataset_type):
		self.split = split
		self.maxx = maxx
		assert self.split in {'TRAIN', 'VAL', 'TEST'}
		 # Load encoded captions (completely into memory)
		with open(os.path.join(data_folder, self.split + '_CAPTIONS' + dataset_type + '.json'), 'r') as j:
			self.captions = json.load(j)

		# Load caption lengths (completely into memory)
		with open(os.path.join(data_folder, self.split + '_CAPLENS' + dataset_type + '.json'), 'r') as j:
			self.caplens = json.load(j)

		# add _x1_IDXS.txt for extra hard (diff 1) img pairs
		with open(os.path.join(data_folder, self.split + "_" + SIMPLICITY + '_IDXS.txt'), 'r') as fp:
			self.fidx = fp.readlines()

		self.imgs = torch.load(os.path.join(data_folder, self.split + "_IMGS.pt"), map_location='cpu')
		self.dataset_size = len(self.fidx)
		
		self.word_map = word_map


	def __len__(self):
		return self.dataset_size

	def getImgEmb(self, img):
		numobj = img.shape[0]

		# if not enough, pad, else random choose some
		if img.shape[0] < self.maxx:
			img = F.pad(img, (0,0,0,self.maxx-img.shape[0]))
		else:
			indices = torch.randint(0, img.shape[0], (self.maxx,))
			img = torch.index_select(img, 0, indices)

		cords = torch.index_select(img, 1, torch.tensor([0, 1, 2, 3])) # (maxx, 4)
		objidx = torch.index_select(img, 1, torch.tensor([5])) # (maxx, 1)

		embs = []
		for i in range(self.maxx):
			if i < numobj:
				word = idx2v[int(objidx[int(i)])]
				embs.append(bert_emb[self.word_map[word]])
			else:
				embs.append(torch.zeros(emb_dim))
		
		embs = torch.stack(embs)
		img = torch.cat((embs, cords), 1)
		return img

	def __getitem__(self, index):
		i = int(self.fidx[index].split('\t')[0])
		j = int(self.fidx[index].split('\t')[1])
		ls = int(self.fidx[index].split('\t')[2])


		#it = random.randint(0, len(self.captions[i])-1)
		img = self.imgs[i]
		# caption = torch.Tensor(self.captions[index][it])
		# caplen = torch.Tensor([self.caplens[index][it]])
		caption = torch.Tensor(self.captions[i])
		caplen = torch.Tensor(self.caplens[i])
		img = self.getImgEmb(img)

		######## distractor
		img2 = self.imgs[j]
		img2 = self.getImgEmb(img2)

		ix = random.randint(0, 1)

		if ix == 0:
			return img, img2, caption, caplen, ix, ls
		else:
			return img2, img, caption, caplen, ix, ls


