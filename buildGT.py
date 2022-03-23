
import os
import re
import json
import click
import string
import torch
import nltk
nltk.download('punkt')
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

from yolo import detect


def buildCaption(disparity, inPath, outPath):    
	# Read origional captions
	lines = []
	with open(inPath+"SimpleSentences1_10020.txt", 'r') as f:
		for line in f:
			if line != '\n':
				lines.append(line[:-1])

	### case 1: hypernym
	if disparity == 'hypernym':
		with open(os.path.join(inPath+'hirc.json'), 'r') as j:
			hirc = json.load(j)
		for k, v in hirc.items():
			for i in range(len(lines)):
				lines[i] = lines[i].replace(k, v)

	### case 2: limited visual
	elif disparity == 'catog':
		rbear = re.compile(r"\bbear\b", re.IGNORECASE)
		rcat = re.compile(r"\bcat\b", re.IGNORECASE)
		rdog = re.compile(r"\bdog\b", re.IGNORECASE)
		rduck = re.compile(r"\bduck\b", re.IGNORECASE)
		rowl = re.compile(r"\bowl\b", re.IGNORECASE)
		rsnake = re.compile(r"\bsnake\b", re.IGNORECASE)
		ranimal = re.compile(r"\banimals\b", re.IGNORECASE)

		rbears = re.compile(r"\bbears\b", re.IGNORECASE)
		rcats = re.compile(r"\bcats\b", re.IGNORECASE)
		rdogs = re.compile(r"\bdogs\b", re.IGNORECASE)
		rducks = re.compile(r"\bducks\b", re.IGNORECASE)
		rowls = re.compile(r"\bowls\b", re.IGNORECASE)
		rsnakes = re.compile(r"\bsnakes\b", re.IGNORECASE)

		for i in range(len(lines)):
			lines[i] = rbear.sub("<unk>", lines[i])
			lines[i] = rcat.sub("<unk>", lines[i])
			lines[i] = rdog.sub("<unk>", lines[i])
			lines[i] = rduck.sub("<unk>", lines[i])
			lines[i] = rowl.sub("<unk>", lines[i])
			lines[i] = rsnake.sub("<unk>", lines[i])
			lines[i] = ranimal.sub("<unk>", lines[i])
			
			lines[i] = rbears.sub("<unk>", lines[i])
			lines[i] = rcats.sub("<unk>", lines[i])
			lines[i] = rdogs.sub("<unk>", lines[i])
			lines[i] = rducks.sub("<unk>", lines[i])
			lines[i] = rowls.sub("<unk>", lines[i])
			lines[i] = rsnakes.sub("<unk>", lines[i])
		
	# write output to file
	with open(outPath+'SimpleSentences1_10020_'+disparity+'.txt', 'w') as filehandle:
		for listitem in lines:
			filehandle.write('%s\n' % listitem)


'''
	Assemble train/validation/test dataset for listener
'''
def assembData_single(disparity, inPath, outPath, image_folder, max_len):
	train_image_paths = []
	train_image_captions = []
	val_image_paths = []
	val_image_captions = []
	test_image_paths = []
	test_image_captions = []
	all_image_paths = []
	all_image_captions = []

	### Read input file
	oldimg = 0
	idx = 0
	captions = []
	with open(os.path.join(inPath, "SimpleSentences1_10020_" + disparity + ".txt"), 'r') as f:
		for line in f:
			idx += 1
			if int(line.split('\t')[0]) != oldimg:
				gp_idx = int(oldimg / 10)
				ig_idx = int(oldimg % 10)
				assert((gp_idx * 10 + ig_idx) == oldimg)
				path = os.path.join(image_folder, "Scene" + str(gp_idx) + "_" + str(ig_idx)+ ".png")
				all_image_paths.append(path)
				all_image_captions.append(captions[:3])

				captions = []
				oldimg = int(line.split('\t')[0])

			cp = line.split('\t')[2]
			cp = cp.translate(str.maketrans('', '', string.punctuation)).lower()
			cp = nltk.word_tokenize(cp)
			captions.append(cp)
				
	gp_idx = int(oldimg / 10)
	ig_idx = int(oldimg % 10)
	assert((gp_idx * 10 + ig_idx) == oldimg)
	path = os.path.join(image_folder, "Scene" + str(gp_idx) + "_" + str(ig_idx)+ ".png")
	all_image_paths.append(path)
	all_image_captions.append(captions)

	print(len(all_image_paths))
	print(len(all_image_captions)) 

	# split into TRAIN/VAL/TES
	train_image_paths = all_image_paths[:8016]
	train_image_captions = all_image_captions[:8016]
	val_image_paths = all_image_paths[8016:9016]
	val_image_captions = all_image_captions[8016:9016]
	test_image_paths = all_image_paths[9016:]
	test_image_captions = all_image_captions[9016:]


	### Output IMG, CAPTIONS, CAPLENS per split
	for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
								(val_image_paths, val_image_captions, 'VAL'),
								(test_image_paths, test_image_captions, 'TEST')]:
		print(len(impaths))
		print(len(imcaps))
		print(split)
		print()
		
		imgs = []
		captions = []
		caplens = []
		
		for i in range(len(impaths)):
			print(impaths[i])
			img = detect(impaths[i], "best.pt")        
			capi = []
			capli = []
			
			for c in imcaps[i]:
				enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
					word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
				c_len = len(c) + 2
				capi.append(enc_c)
				capli.append(c_len)
			
			imgs.append(img)
			captions.append(capi)
			caplens.append(capli)
		
		with open(os.path.join(outPath, split + '_CAPTIONS_' + disparity + '.json'), 'w') as j:
			json.dump(captions, j)

		with open(os.path.join(outPath, split + '_CAPLENS_' + disparity + '.json'), 'w') as j:
			json.dump(caplens, j)
		
		torch.save(imgs, os.path.join(outPath, split+'_IMGS.pt'))


'''
	Assemble train/validation/test dataset for speaker (origional+disparity)
'''
def assembData_double(disparity, inPath, outPath, image_folder, max_len):
	train_image_paths = []
	train_image_captions = []
	val_image_paths = []
	val_image_captions = []
	test_image_paths = []
	test_image_captions = []
	all_image_paths = []
	all_image_captions = []

	oldimg = 0
	num_rec = 0
	captions = []
	with open(os.path.join(root_path, "SimpleSentences1_10020.txt"), 'r') as f, \
		open(os.path.join(root_path, "SimpleSentences1_10020_"+disparity+".txt"), 'r') as f2: 
		
		for x, y in zip(f, f2):
			assert(x.split('\t')[0:2] == y.split('\t')[0:2])
			if int(x.split('\t')[0]) != oldimg:
				gp_idx = int(oldimg / 10)
				ig_idx = int(oldimg % 10)
				assert((gp_idx * 10 + ig_idx) == oldimg)
				path = os.path.join(image_folder, "Scene" + str(gp_idx) + "_" + str(ig_idx)+ ".png")
				all_image_paths.append(path)
				all_image_captions.append(captions[:6])

				captions = []
				oldimg = int(x.split('\t')[0])

			cp = x.split('\t')[2]
			cp = cp.translate(str.maketrans('', '', string.punctuation)).lower()
			cp = nltk.word_tokenize(cp)
			word_freq.update(cp)
			captions.append(cp)
			
			cp = y.split('\t')[2]
			cp = cp.translate(str.maketrans('', '', string.punctuation)).lower()
			cp = nltk.word_tokenize(cp)
			word_freq.update(cp)
			captions.append(cp)
				
	gp_idx = int(oldimg / 10)
	ig_idx = int(oldimg % 10)
	assert((gp_idx * 10 + ig_idx) == oldimg)
	path = os.path.join(image_folder, "Scene" + str(gp_idx) + "_" + str(ig_idx)+ ".png")
	all_image_paths.append(path)
	all_image_captions.append(captions)

	print(len(all_image_paths))
	print(len(all_image_captions)) 

	train_image_paths = all_image_paths[:8016]
	train_image_captions = all_image_captions[:8016]
	val_image_paths = all_image_paths[8016:9016]
	val_image_captions = all_image_captions[8016:9016]
	test_image_paths = all_image_paths[9016:]
	test_image_captions = all_image_captions[9016:]

	for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
								(val_image_paths, val_image_captions, 'VAL'),
								(test_image_paths, test_image_captions, 'TEST')]:
		imgs = []
		captions = []
		caplens = []
		
		for i in range(len(impaths)):
			img = detect(impaths[i], "best.pt")        
			capi = []
			capli = []
			
			if i%500 == 0:
				print(i)
			
			for c in imcaps[i]:
				enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
					word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
				c_len = len(c) + 2
				capi.append(enc_c)
				capli.append(c_len)
			
			imgs.append(img)
			captions.append(capi)
			caplens.append(capli)
		
		with open(os.path.join(outPath, split + '_CAPTIONS_n_' + disparity + '.json'), 'w') as j:
			json.dump(captions, j)

		with open(os.path.join(outPath, split + '_CAPLENS_n_' + disparity + '.json'), 'w') as j:
			json.dump(caplens, j)
		
		torch.save(imgs, os.path.join(outPath, split+'_IMGS.pt'))


'''
	Embed wordmap with pre-trained BERT embedding
	Dimension changes according to wordmap size
'''
def BERTemb(disparity, inPath, outPath):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-uncased')
	model.eval()

	with open(inPath + "wordmap_" + disparity + ".json", 'r') as j:
		word_map = json.load(j)
		rev_word_map = {v: k for k, v in word_map.items()} 

	print(len(word_map))
	print(len(rev_word_map))

	wmtrx = np.zeros((len(word_map), 768))
	for i, (k, v) in enumerate(rev_word_map.items()):
		if v == '<unk>':
			v = '[UNK]'
		elif v == '<start>':
			v = '[CLS]'
		elif v == '<end>':
			v = '[SEP]'
		elif v == '<pad>':
			v = '[PAD]'
		tokenized_cap = tokenizer.tokenize(v)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
		tokens_tensor = torch.tensor([indexed_tokens])
		encoded_layers, _ = model(tokens_tensor)
		bert_embedding = encoded_layers[11].squeeze(0)
		if bert_embedding.shape[0] != 1:
			if tokenized_cap[1] == '_':
				undscor = bert_embedding[1]
			bert_embedding = bert_embedding.sum(dim=0, keepdim=True)
			if tokenized_cap[1] == '_':
				bert_embedding = torch.sub(bert_embedding, undscor)
		wmtrx[i] = bert_embedding.detach().numpy()
	print(wmtrx.shape)

	pickle.dump(wmtrx, open(outPath + 'BERT_EMB_' + disparity + '.pkl', 'wb'), protocol=2)




@click.command()
@click.option('--disparity', '-d', help='Disparity type: hypernym, catog')
@click.option('--inpath', '-i', default='input/', help='The input file path')
@click.option('--outpath', '-o', default='input/', help='The output file path')
@click.option('--imgpath', '-img', default='AbstractScenes_v1.1/RenderedScenes/', help='Input image file folder')
@click.option('--maxlen', '-l', default=25, help='max sentence length')

def main(disparity, inpath, outpath, imgpath, maxlen):
	# build caption and BERT embedding
	buildCaption(disparity, inpath, outpath)
	BERTemb(disparity, inpath, outpath)

	# build train/val/test datasets
	if disparity == 'hypernym':
		# listener
		buildCaption(disparity, inpath, outpath)
		BERTemb(disparity, inpath, outpath)
		assembData_single(disparity, inpath, outpath, imgpath, maxlen) 

		# speaker
		buildCaption('n_hypernym', inpath, outpath)
		BERTemb('n_hypernym', inpath, outpath)
		assembData_double(disparity, inpath, outpath, imgpath, maxlen)

	elif disparity == 'catog':
		# speaker is the origional caption, only needs to build listener
		buildCaption(disparity, inpath, outpath)
		BERTemb(disparity, inpath, outpath)
		assembData_single(disparity, inpath, outpath, imgpath, maxlen)
		

	print("### Data Preprocessing Complete!")



if __name__ == '__main__':
	main()






