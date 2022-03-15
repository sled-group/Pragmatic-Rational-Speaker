import torch
import torch.nn.functional as F
from scipy import spatial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pickone(decoder, word_map, img1, img2, caps, caplens):

	img1 = img1.to(device).float()
	img2 = img2.to(device).float()
	caps = caps.to(device).long()
	caplens = caplens.to(device).squeeze(1).long()


	# Forward prop.
	score1 = decoder(img1, caps, caplens) #(batch, len, vocab_size)
	score2 = decoder(img2, caps, caplens)

	sc1 = score1.argmax(dim=2) #(batch, len)
	sc2 = score2.argmax(dim=2)

	# Clean up sentences, get the clean caption of each (batch, sentlen)
	hp1 = list(map(lambda c: [w for w in c if w not in {word_map['<end>'], word_map['<start>'], word_map['<pad>']}],
					sc1.to('cpu').numpy()))
	hp2 = list(map(lambda c: [w for w in c if w not in {word_map['<end>'],word_map['<start>'], word_map['<pad>']}],
					sc2.to('cpu').numpy()))
	cp = list(map(lambda c: [w for w in c if w not in {word_map['<end>'], \
														word_map['<start>'], \
														word_map['<pad>']}],caps.to('cpu').numpy()))    

	idxx = []
	diff = []
	for i in range(img1.shape[0]): # batch
		# encode every clean word and sum the embedding
		cpi = decoder.embedding(torch.tensor(cp[i]).long().to(device)).sum(dim=0).cpu().detach().numpy()
		sc1i = decoder.embedding(torch.tensor(hp1[i]).long().to(device)).sum(dim=0).cpu().detach().numpy()
		sc2i = decoder.embedding(torch.tensor(hp2[i]).long().to(device)).sum(dim=0).cpu().detach().numpy()
		
		# similarity, higher better
		res1 = 1 - spatial.distance.cosine(sc1i, cpi)
		res2 = 1 - spatial.distance.cosine(sc2i, cpi)
		it = (res1 < res2) * 1.0

		idxx.append(it)
		if it == 0:
			diff.append(res1 - res2)
		else:
			diff.append(res2 - res1)

	return idxx, diff