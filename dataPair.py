import os
import json
import torch
import pickle
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset



class MyDatasetPair(torch.utils.data.Dataset):
	def __init__(self, data_folder, split, maxx, emb_dim, bert_emb, word_map, disparity, SIMPLICITY):
		self.split = split
		self.maxx = maxx
		self.emb_dim = emb_dim
		self.word_map = word_map
		self.bert_emb = bert_emb
		self.disparity = disparity

		if disparity == 'hypernym':
			type_speaker = '_n_hypernym'
			type_listener = '_hypernym'
		else:
			type_speaker = ''
			type_listener = '_catog'

		assert self.split in {'TRAIN', 'VAL', 'TEST'}
		 # Load encoded captions (completely into memory)
		with open(os.path.join(data_folder, self.split + '_CAPTIONS' + type_speaker + '.json'), 'r') as j:
			self.captions = json.load(j)

		# Load caption lengths (completely into memory)
		with open(os.path.join(data_folder, self.split + '_CAPLENS' + type_speaker + '.json'), 'r') as j:
			self.caplens = json.load(j)

		# add _x1_IDXS.txt for extra hard (diff 1) img pairs
		with open(os.path.join(data_folder, self.split + "_" + SIMPLICITY + '_IDXS.txt'), 'r') as fp:
			self.fidx = fp.readlines()

		self.imgs = torch.load(os.path.join(data_folder, self.split + "_IMGS.pt"), map_location='cpu')
		self.dataset_size = len(self.fidx)
		
		# Read obj detection category
		with open(os.path.join(data_folder, 'v2idx.json'), 'r') as j:
			v2idx = json.load(j)
		self.idx2v = {v:k for k,v in v2idx.items()}

		if disparity == 'catog':
			bert_emb_l = pickle.load(open(os.path.join(data_folder, 'BERT_EMB' + type_listener + '.pkl'), 'rb'))
			self.bert_emb_l = torch.tensor(bert_emb_l)
			with open(os.path.join('input/', 'v2idx' + type_listener + '.json'), 'r') as j:
				self.idx2v_l = json.load(j)
			with open(os.path.join(data_folder, 'wordmap' + type_listener + '.json'), 'r') as j:
				self.word_map_l = json.load(j)



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
				word = self.idx2v[int(objidx[int(i)])]
				embs.append(self.bert_emb[self.word_map[word]])
			else:
				embs.append(torch.zeros(self.emb_dim))
		
		embs = torch.stack(embs)
		img = torch.cat((embs, cords), 1)
		
		if self.disparity == 'hypernym':
			return img, img

		else:
			embs_l = []
			for i in range(self.maxx):
				if i < numobj:
					word = self.idx2v_l[str(int(objidx[int(i)]))]
					embs_l.append(self.bert_emb_l[self.word_map_l[word]])
				else:
					embs_l.append(torch.zeros(self.emb_dim))
			
			embs_l = torch.stack(embs_l)
			img_l = torch.cat((embs_l, cords), 1)
			return img, img_l


	def __getitem__(self, index):
		i = int(self.fidx[index].split('\t')[0])
		j = int(self.fidx[index].split('\t')[1])
		ls = int(self.fidx[index].split('\t')[2])

		img = self.imgs[i]
		caption = torch.Tensor(self.captions[i])
		caplen = torch.Tensor(self.caplens[i])
		img2 = self.imgs[j]
		ix = random.randint(0, 1)

		if self.disparity == 'hypernym':
			img, _ = self.getImgEmb(img)
			img2, _ = self.getImgEmb(img2)
			if ix == 0:
				return img, img2, img, img2, caption, caplen, ix, ls
			else:
				return img2, img, img2, img, caption, caplen, ix, ls

		else:
			img, img_l = self.getImgEmb(img)
			img2, img2_l = self.getImgEmb(img2)
			if ix == 0:
				return img, img2, img_l, img2_l, caption, caplen, ix, ls
			else:
				return img2, img, img2_l, img_l, caption, caplen, ix, ls

		