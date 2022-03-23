import torch
import torch.nn as nn

class Policy_la(nn.Module):
	def __init__(self, decoder_dim, out_dim, beam_size, vocab_size, bert_emb):
		super(Policy_la, self).__init__()
		self.out_dim = out_dim
		self.embedding = nn.Embedding(vocab_size, decoder_dim)
		self.embedding.weight = nn.Parameter(bert_emb)
		for p in self.embedding.parameters():
			p.requires_grad = True

		self.relu = nn.ReLU()
		self.logsftx = nn.LogSoftmax(dim=1)
		self.output = nn.Linear(decoder_dim, 1)


	@staticmethod
	def _init_weights(module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=0.2)


	'''
		Forward inputs:
			captions: (batch, beam, seqlen)
			caption_lengths: (batch, beam, 1)
			logs, idall, dfall: (batch, beam)
	'''
	def forward(self, captions, caption_lengths, logs, idall, dfall, ix):
		ones = torch.ones_like(captions)
		caption_mask = caption_lengths > ones.cumsum(dim=2)
		caption_embeddings = self.embedding(caption_mask * captions).float().sum(dim=2)

		idfall = dfall*(idall==ix.unsqueeze(1))
		scores = self.output(caption_embeddings).squeeze(-1) #(batch, beam, outdim)
		scores = scores.pow(l_p) * idfall.pow(l_c)
		scores = self.logsftx(scores)

		#(batch, beam_size)
		return scores

