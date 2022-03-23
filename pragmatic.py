import os
import json
import click
import torch
import torch.nn as nn
import torch.optim as optim
from statistics import mean

from utils.train import *
from dataPair import *
from speaker import speaker, pickone
from model import Policy_la

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Hyperparameter initialization & placeholder
REPEAT = 1
TESTIME = False
SIMPLICITY = 'b' # both: b, hard: h, simple: s
disparity = 'hypernym'

ckpt = None
data_folder = 'input/'


bert_emb = None
word_map = None
rev_word_map = None
word_map_l = None

decoder = None
decoder_l = None

### training parameters
epochs = 20
learning_rate = 1e-4
batch_size = 128


### policy model parameters
out_dim = 64
decoder_dim = 768
vocab_size = 0 # len(word_map)

## speaker beam search
maxx = 9
beam_size = 30
emb_dim = 768
max_dec_step = 30
# hypernym: (1337, 1338); catog: (1086, 1087)
start_idx = 0 
end_idx = 0


'''
	change from speaker word_map idx to listener word_map idx
	act: (batch, length), cpln: (batch)
'''
def changeDic(act):
	cpln = []
	for i in range(act.shape[0]):
		cpln.append(act.shape[1])
		for j in range(act.shape[1]):
			# switch words
			word = rev_word_map[act[i][j].item()]
			if word in word_map_l:
				act[i][j] = word_map_l[word]
			else:
				act[i][j] = word_map_l['<unk>']
			# find end token
			if (word == '<end>') and (cpln[i] == act.shape[1]):
				cpln[i] = j + 1

	return act, cpln




@torch.no_grad()
def test(policy, split):
	# Data Loader
	train_loader = torch.utils.data.DataLoader(
		MyDatasetPair(data_folder, split, maxx, emb_dim, bert_emb, word_map, \
			disparity, SIMPLICITY), batch_size=batch_size, shuffle=True, \
			num_workers=1, pin_memory=True)

	policy.eval()
	tot_acc = 0
	tot_siz = 0
	bins = {}

	# Run tests
	for i, (img1, img2, img1_l, img2_l, caps, caplens, ix, ls) in enumerate(train_loader):
		cur_batch = img1.shape[0]
		img1 = img1.to(device).float()
		img2 = img2.to(device).float()
		ix = ix.to(device)

		# (batch, beam, len) for seq, 
		# (batch, beam) for cpln (1), logs, idall, dfall
		seq, cpln, logs, idall, dfall = speaker(decoder, word_map, img1, img2, \
										ix, maxx, beam_size, emb_dim, \
										max_dec_step, start_idx, end_idx)

		# compute a candidate score
		probs = policy(seq, cpln, logs, idall, dfall, ix) #(batch, beam)
		mxprob, act_idx = probs.max(dim=1) #(batch_size,)

		### gather origional sentence
		seq = seq.view(cur_batch, beam_size, seq.shape[-1])
		act_idx = act_idx.unsqueeze(1)
		act_idx = act_idx.expand(-1, seq.shape[-1])
		act_idx = act_idx.unsqueeze(1)
		act = torch.gather(seq, 1, act_idx).squeeze(1)


		### Vocab rematch, find cpln
		act = act.to('cpu')
		act, cpln = changeDic(act)
		act = act.to(device) #(batch, len)
		cpln = torch.tensor(cpln).unsqueeze(1).long().to(device)

		if disparity == 'catog':
			### get new image for listener
			img1_l = img1_l.to(device).float()
			img2_l = img2_l.to(device).float()

			### Send it to listener
			idxx_l, diff_l = pickone(decoder_l,word_map_l, img1_l, \
									img2_l, act, cpln) #(batch, )

		else:	
			### Send it to listener
			idxx_l, diff_l = pickone(decoder_l,word_map_l, img1, img2, \
									act, cpln) #(batch, )

		tmp_acc = 1.0*(torch.tensor(idxx_l).to(device) == ix)
		for k,v in zip(ls.tolist(), tmp_acc.tolist()):
			if k not in bins:
				bins[k] = []
			bins[k].append(v)

		reward = 1.0*(torch.tensor(idxx_l).to(device) == ix).sum().to(device)

		tot_acc += reward
		tot_siz += cur_batch

	return tot_acc/tot_siz



@click.command()
@click.option('--disparityin', '-d', help='Disparity type: hypernym, catog')
@click.option('--simplicity', '-s', default='b', help='Simplicity of the dataset, b: both, s: simple, h: hard')
@click.option('--testime', '-t', default=False, help='Train or Test mode, default Train')
@click.option('--repeat', '-r', default=1, help='Number of tests to repeat, default 1')
@click.option('--inpath', '-i', default='input/', help='The input file path')
@click.option('--ckptin', '-c', default=None, help='Checkpoint')

def main(disparityin, simplicity, testime, repeat, inpath, ckptin):	
	global disparity, SIMPLICITY, TESTIME, REPEAT, data_folder, start_idx, \
			end_idx, ckpt, bert_emb, word_map, rev_word_map, word_map_l, \
			vocab_size, decoder, decoder_l

	disparity = disparityin
	SIMPLICITY = simplicity
	TESTIME = testime
	REPEAT = repeat
	data_folder = inpath
	ckpt = ckptin

	if disparity == 'hypernym':
		type_speaker = '_n_hypernym'
		type_listener = '_hypernym'
		start_idx = 1337
		end_idx = 1338
	else:
		type_speaker = ''
		type_listener = '_catog'
		start_idx = 1086
		end_idx = 1087

	## load speaker inputs
	checkpoint_s = torch.load('ckpts/BEST_checkpoint_caption' + 
				type_speaker +'.pth.tar', map_location=str(device))
	decoder = checkpoint_s['decoder']
	decoder = decoder.to(device)
	decoder.eval()

	bert_emb = pickle.load(open(data_folder + 'BERT_EMB' + type_speaker + '.pkl', 'rb'))
	bert_emb = torch.tensor(bert_emb).to(device)
	with open(os.path.join(data_folder, 'wordmap' + type_speaker + '.json'), 'r') as j:
		word_map = json.load(j)
	rev_word_map = {v: k for k, v in word_map.items()}
	vocab_size = len(word_map)


	## load listener inputs
	checkpoint_l = torch.load('ckpts/BEST_checkpoint_caption' + 
					type_listener + '.pth.tar', map_location=str(device))
	decoder_l = checkpoint_l['decoder']
	decoder_l = decoder_l.to(device)
	decoder_l.eval()

	with open(os.path.join(data_folder, 'wordmap' + type_listener + '.json'), 'r') as j:
		word_map_l = json.load(j)
	rev_word_map_l = {v: k for k, v in word_map_l.items()}

	
	# Initialization
	if ckpt is None:
		policy = Policy_la(decoder_dim=decoder_dim, out_dim=out_dim, \
						beam_size=beam_size, vocab_size = vocab_size, \
						bert_emb = bert_emb).to(device)
		optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
		running_reward = 0
		best_reward = 0
		epochs_since_improvement = 0
	else:
		checkpoint = torch.load(ckpt)
		start_epoch = checkpoint['epoch'] + 1
		epochs_since_improvement = checkpoint['epochs_since_improvement']
		best_reward = checkpoint['bleu-4']
		running_reward = best_reward
		policy = checkpoint['decoder']
		optimizer = checkpoint['decoder_optimizer']

	if TESTIME:
		for ri in range(REPEAT):
			test(policy, "TEST")
		return


	# Load Data
	train_loader = torch.utils.data.DataLoader(
		MyDatasetPair(data_folder, 'TRAIN', maxx, emb_dim, bert_emb, word_map, \
			disparity, SIMPLICITY), batch_size=batch_size, shuffle=True, \
			num_workers=1, pin_memory=True)

	### start training
	for e in range(epochs):
		# Adjust learning rate
		if epochs_since_improvement > 0 and epochs_since_improvement % 50 == 0:
			adjust_learning_rate(optimizer, 0.8)

		for i, (img1, img2, img1_l, img2_l, caps, caplens, ix, ls) in enumerate(train_loader):
			cur_batch = img1.shape[0]
			img1 = img1.to(device).float()
			img2 = img2.to(device).float()
			ix = ix.to(device)

			# rational speaker (image caption + simulate listener)
			with torch.no_grad():
				# (batch, beam, len) for seq, 
				# (batch, beam) for cpln (1), logs, idall, dfall
				seq, cpln, logs, idall, dfall = speaker(decoder, word_map, img1,\
										img2, ix, maxx, beam_size, emb_dim, \
										max_dec_step, start_idx, end_idx)


			# compute a candidate score
			probs = policy(seq, cpln, logs, idall, dfall, ix) #(batch, beam)
			mxprob, act_idx = probs.max(dim=1) #(batch_size,)

			### send to listener
			with torch.no_grad():
				### gather origional sentence
				act_idx = act_idx.unsqueeze(1)
				act_idx = act_idx.expand(-1, seq.shape[-1])
				act_idx = act_idx.unsqueeze(1)
				act = torch.gather(seq, 1, act_idx).squeeze(1)

				### Vocab rematch, find cpln
				act = act.to('cpu')
				act, cpln = changeDic(act)
				act = act.to(device) #(batch, len)
				cpln = torch.tensor(cpln).unsqueeze(1).long().to(device)

				if disparity == 'catog':
					### get new image for listener
					img1_l = img1_l.to(device).float()
					img2_l = img2_l.to(device).float()

					### Send it to listener
					idxx_l, diff_l = pickone(decoder_l,word_map_l, img1_l, \
											img2_l, act, cpln) #(batch, )

				else:	
					### Send it to listener
					idxx_l, diff_l = pickone(decoder_l,word_map_l, img1, img2, \
											act, cpln) #(batch, )

			# Compute loss
			reward = (torch.tensor(idxx_l).to(device) == ix)*2-1
			optimizer.zero_grad()
			loss = (0.001+mxprob)*(reward==1)*reward + \
					(reward==-1)*(torch.log(0.001+1-mxprob.exp()))
			loss = -loss.mean()
			loss.backward()
			optimizer.step()

			# Calculate reward
			reward = reward.sum()/cur_batch
			running_reward = 0.005 * reward + (1 - 0.005) * running_reward

		# Evaluation
		testacc = test(policy, 'VAL').to(device)
		is_best = testacc > best_reward
		best_reward = max(testacc, best_reward)
		if not is_best:
			epochs_since_improvement += 1
			print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
		else:
			epochs_since_improvement = 0

		# Save checkpoint
		save_checkpoint("ckpts/", disparity+"_"+SIMPLICITY, e, \
						epochs_since_improvement, policy, optimizer, \
						running_reward, is_best)




if __name__ == '__main__':
	main()

