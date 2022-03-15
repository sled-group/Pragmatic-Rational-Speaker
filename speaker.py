import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trstep(decoder, img, partial_captions, beam_size, emb_dim, maxx):
	# partial_captions: (batch_size * beam_size, timesteps)
	# img: (batch, maxx, 772)
	cur_batch = img.shape[0]
	img = img.unsqueeze(1).repeat(1, beam_size, 1, 1)
	img = img.view(cur_batch * beam_size, maxx, emb_dim+4)  # (batch*beam, 9, 772)

	caption_lengths = torch.ones_like(partial_captions,)  # (batch,)
	if len(caption_lengths.size()) == 2:
		caption_lengths = caption_lengths.sum(1)
	else:
		# Add a time-step. shape: (batch_size, 1)
		partial_captions = partial_captions.unsqueeze(1)
	# (batch*beam, partial_length, vocab_size)
	output_logits = decoder(img, partial_captions, caption_lengths)

	# Keep features for last time-step only
	output_logits = output_logits[:, -1, :]
	next_logprobs = F.log_softmax(output_logits, dim=1)  # (batch*beam, vocab_size)

	# Set logprobs of last predicted tokens as high negative value to avoid
	# repetition in caption.
	for index in range(partial_captions.shape[0]):
		next_logprobs[index, partial_captions[index, -1]] = -10000

	return next_logprobs


def caption_image_beam_search(decoder, img, beam_size, start_idx, end_idx, maxx,
							emb_dim, vocab_size, max_dec_step):
	# img: (batch, 9, 772)
	cur_batch = img.shape[0]
	predictions: List[torch.Tensor] = []  # (batch_size, beam_size), no start
	backpointers: List[torch.Tensor] = []  # (batch_size, beam_size), parent idx, no start

	start_predictions = img.new_full((cur_batch,), start_idx).long() #(batch, )
	start_class_log_probs = trstep(decoder, img, start_predictions, beam_size, emb_dim, maxx)  # (batch*beam, vocab_size)

	# shape: (batch_size, beam_size), (batch_size, beam_size)
	start_top_log_probs, start_predicted_classes = start_class_log_probs.topk(beam_size)
	last_log_probs = start_top_log_probs  # (batch_size, beam_size)) .permute(1, 0)
	predictions.append(start_predicted_classes)
	log_probs_after_end = start_class_log_probs.new_full(
						(cur_batch * beam_size, vocab_size), float("-inf"))
	log_probs_after_end[:, end_idx] = 0.0

	for timestep in range(max_dec_step - 1):
		last_predictions = predictions[-1].reshape((cur_batch * beam_size,))  # (batch_size * beam_size,)
		if (last_predictions == end_idx).all():
			break

		predictions_so_far = torch.stack(predictions).permute(1, 2, 0).view(cur_batch * beam_size, -1)
		class_log_probs = trstep(decoder, img, predictions_so_far, beam_size, emb_dim, maxx)  # (batch_size * beam_size, num_classes)		
		last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
				cur_batch * beam_size, vocab_size)  # (batch_size * beam_size, num_classes)
		

		cleaned_log_probs = torch.where(
				last_predictions_expanded == end_idx,
				log_probs_after_end,
				class_log_probs,)
		top_log_probs, predicted_classes = cleaned_log_probs.topk(beam_size)  # (ba*be, be)
		expanded_last_log_probs = (
				last_log_probs.unsqueeze(2)
				.expand(cur_batch, beam_size, beam_size)
				.reshape(cur_batch * beam_size, beam_size))

		# sum to get all prob
		summed_top_log_probs = top_log_probs + expanded_last_log_probs
		reshaped_summed = summed_top_log_probs.reshape(
				cur_batch, beam_size * beam_size)
		reshaped_predicted_classes = predicted_classes.reshape(
				cur_batch, beam_size * beam_size)
		
		# Keep only the top `beam_size` beam indices.
		# shape: (batch_size, beam_size), (batch_size, beam_size)
		restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(beam_size)

		# Use the beam indices to extract the corresponding classes.
		# shape: (batch_size, beam_size)
		restricted_predicted_classes = reshaped_predicted_classes.gather(
			1, restricted_beam_indices)
		predictions.append(restricted_predicted_classes)

		# shape: (batch_size, beam_size)
		last_log_probs = restricted_beam_log_probs

		# shape: (batch_size, beam_size)
		backpointer = restricted_beam_indices // beam_size
		backpointers.append(backpointer)

	# get cap lens of each
	all_caplen = torch.full_like(last_log_probs, len(predictions)).unsqueeze(-1)

	# Reconstruct the captions.
	# shape: [(batch_size, beam_size, 1)]
	reconstructed_predictions = [predictions[-1].unsqueeze(2)]

	# shape: (batch_size, beam_size)
	cur_backpointers = backpointers[-1]

	for timestep in range(len(predictions) - 2, 0, -1):
		# shape: (batch_size, beam_size, 1)
		cur_preds = (
			predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
		)
		reconstructed_predictions.append(cur_preds)
		all_caplen = (cur_preds!=end_idx)*1.0*all_caplen + (cur_preds==end_idx)*1.0*(timestep+1)

		# shape: (batch_size, beam_size)
		cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

	# shape: (batch_size, beam_size, 1)
	final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
	all_caplen = (final_preds!=end_idx)*1.0*all_caplen + (final_preds==end_idx)*1.0*(timestep+1)

	reconstructed_predictions.append(final_preds)

	# shape: (batch_size, beam_size, max_steps)
	all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
	
	# returning logs, could be helpful for ranking, all_calpn add start
	return all_predictions, last_log_probs, all_caplen+1


@torch.no_grad()
def speaker(decoder, word_map, img1, img2, ix, maxx, beam_size, emb_dim, \
			max_dec_step, start_idx, end_idx):

	cur_batch = img1.shape[0]
	vocab_size=len(word_map)
	img1 = img1.to(device).float()
	img2 = img2.to(device).float()

	### generate target imgs
	img0 = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)))  # (2, batch, maxx, 772)
	midx = ix.unsqueeze(0) #(1, cur_batch)
	midx = midx.unsqueeze(-1)
	midx = midx.expand(1, cur_batch, img1.shape[-2])
	midx = midx.unsqueeze(-1)
	midx = midx.expand(1, cur_batch, img1.shape[-2], img1.shape[-1]).to(device)
	img1s = torch.gather(img0, 0, midx).squeeze(0)  # targeted imgs


	### Target Image Caption
	seq, logs, cpln = caption_image_beam_search(decoder, img1s, beam_size, \
												start_idx, end_idx, maxx, \
												emb_dim, vocab_size, max_dec_step)

	# add <start> to seq
	startpad = torch.full((cur_batch, beam_size), start_idx).to(device)
	seq = torch.cat((startpad.unsqueeze(-1), seq), dim=-1)  # (batch, beam, seqlen)

	### PICKONE Simulating Listener
	seq = seq.permute(1, 0, 2) #(beam, batch, seqlen)
	cpln = cpln.permute(1, 0, 2)
	idall = []  # (beam, batch,)
	dfall = []  # (beam, batch,)
	for si in range(beam_size):
		idxx, diff = pickone(decoder, word_map, img1, img2, seq[si], cpln[si]) #(batch, )
		idall.append(idxx)
		dfall.append(diff)

	
	# Pick the highest one
	idall = torch.tensor(idall).permute(1, 0).to(device)
	dfall = torch.tensor(dfall).permute(1, 0).to(device)
	seq = seq.permute(1, 0, 2).to(device)
	cpln = cpln.permute(1, 0, 2).to(device)

	# (batch, beam, len) for seq, (batch, beam) for cpln (1), logs, idall, dfall
	return seq, cpln, logs, idall, dfall
