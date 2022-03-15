# Pragmatic-Rational-Speaker

This is the Github page for the 2022 ACL paper "Learning to Mediate Disparities Towards Pragmatic Communication". 

![xxx](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/utils/model.png)

## Abstract

Human communication is a collaborative process. Speakers, on top of conveying their own intent, adjust the content and language expressions by taking the listeners into account, including their knowledge background, personality, and physical capabilities. Towards building AI agents that have similar abilities in language communication, we propose a novel rational reasoning framework, Pragmatic Rational Speaker (PRS), where the speaker attempts to learn the speaker-listener disparity and adjust the speech accordingly, by adding a light-weighted disparity adjustment layer into *working memory* on top of speaker’s *long-term* memory system. By fixing the long-term memory, the PRS only needs to update its working memory to learn and adapt to different types of listeners. To validate our framework, we create a dataset that simulates different types of speaker-listener disparities in the context of referential games. Our empirical results demonstrate that the PRS is able to shift its output towards the language that listeners are able to understand, significantly improve the collaborative task outcome, and learn the disparity faster than joint training.


## Dataset

We modified the [Abstract Scenes](http://optimus.cc.gatech.edu/clipart/) (Gilberto Mateos Ortiz et al., 2015) dataset for our experiments. There are 10020 images, each including 3 ground truth captions, and a median of 6 to 7 objects. 

We assembled ∼35k pairs of images that differ by ≤ 4 objects as the Hard set(h), ∼25k pairs that differ by > 4 objects as the Easy set(s), and together as the Combined set(b). The image pairs were split into training, validation and testing by a ratio of 8:1:1.

The paired image dataset can be round in the [input](https://github.com/Anbyew/Pragmatic-Rational-Speaker/tree/main/input) folder: [split]_[difficulty]_IDX.txt.
e.g. TRAIN_s_IDX.txt. Each file includes three columns: {img1_idx, img2_idx, #diff_obj}.

To create disparities, the following files in the [input](https://github.com/Anbyew/Pragmatic-Rational-Speaker/tree/main/input) folder were used to modify the ground truth training dataset for the corresponding listener's' image-caption grounding module: 
- Hypernym: [changeD_hypernym.json](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/input/changdD_hypernym.json)
- Limited Visual: [changdD_catog.json](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/input/changdD_catog.json)


## Models

### Literal and Rational Speaker

The Literal Speaker is an object detection based image captioning module that generates
caption candidates for the target image. 

1. Objection Detection

	We retrained [YOLO v3](https://github.com/ultralytics/yolov3/) from scratch using individual images in the Abstract Scene dataset. The inference time code can be found in [yolo.py](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/yolo.py)

2. Image Captioning

	We adapted the [Show, Attend, and Tell](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) code, retrained the captioning module from scratch using individual images in the Abstract Scene dataset. The inference time code can be found in [speaker.py](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/speaker.py)

3. Internal Listener Simulation

	Without disparity concerns, the Rational Speaker fulfills the task goal by simulating the Rational Listener’s behavior, and rank the candidate captions generated by the Literal Speaker according to how well they can describe the target image apart from the distractors.


### Rational Listener with/without Disparity

Rational Listener picks out the image that they believe is the target. We reuse the same Fixed
pre-trained Training-mode Transformer module to decide which image does the caption ground better in. The model can be found in [listener.py](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/listener.py)

To create listeners with disparities, retrain the image captioning model from previous step using the new dataset for each type of disparity.


### Pragmatic Rational Speaker
On top of the Rational Speaker, the Pragmatic Rational Speaker incorporates a disparity adjustment layer to learn and accommodate the listener’s disparity through REINFORCE interactions. The model can be found in [pragmatic.py](https://github.com/Anbyew/Pragmatic-Rational-Speaker/blob/main/pragmatic.py)



## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For further questions, please contact yuweibao@umich.edu.

