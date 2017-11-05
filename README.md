# C2TutorialsGo
This is a tutorial written for Caffe2 which mocks google AlphaGo Fan and AlphaGO Zero.
v0.1.1 is the currently recommended version if you want stable result.

## Installation
  This program by so far relies on [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) Cython implementation for feature preprocessing and Go rules. Cython compilation can be done by running shell command `python setup.py build_ext --inplace`.

# New updates from AlphaGo Zero
## Preprocess
    The Go game dataset are usually stored in [SGF](http://www.red-bean.com/sgf/go.html) file format. We need to transform SGF file into Caffe2 Tensor. AlphaGo Zero requires 17 feature planes of 19x19 size, which does not include 'human knowledge' like Liberties or Escape.  
    [This preprocess program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20Zero%20%281%29%20Preprocess%20Pipeline.ipynb) still relies on [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) for Go rules, but no more dependencies for feature generation. I'm looking for a better(more accurate) Go rule implementation which can support Chinese/Korean/Japanese Go rules and different Komi, please feel free to recommend.

## Dual Policy and Value network with ResNet  
    The Supervised Learning program is used to evaluate whether the network architecture is correct. It can now run in CPU mode, of course very slow, and meet gradient explosion in GPU mode, I'm trying to locate where the problem is. Welcome to raise pull requests.
    
| epochs | LR     | loss   | train/test accu | epochs | LR     | loss   | train/test accu |
|--------|--------|--------|-----------------|--------|--------|--------|-----------------|
| 1      |        |        |        /        | 11     |        |        |        /        |
| 2      |        |        |        /        | 12     |        |        |        /        |
| 3      |        |        |        /        | 13     |        |        |        /        |
| 4      |        |        |        /        | 14     |        |        |        /        |
| 5      |        |        |        /        | 15     |        |        |        /        |
| 6      |        |        |        /        | 16     |        |        |        /        |
| 7      |        |        |        /        | 17     |        |        |        /        |
| 8      |        |        |        /        | 18     |        |        |        /        |
| 9      |        |        |        /        | 19     |        |        |        /        |
| 10     |        |        |        /        | 25*    |        |        | 0.60/0.57(alphago zero)|

## Reinforced Learning pipline
    On going. This will be different from AlphaGo Fan in may ways:
    1. Always use the best primary player to generate data.
    2. Before each move, do wide search to obtain better distribution than Policy predict.
    3. MCTS only relies on Policy and Value network, no more Rollout.
    4. more detail will be added during implementation

# About AlphaGo Fan
## Preprocess
  The Go game dataset are usually stored in [SGF](http://www.red-bean.com/sgf/go.html) file format. We need to transform SGF file into Caffe2 Tensor which are 48 feature planes of 19x19 size, according to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true).  
    [The preprocess program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%281%29%20Preprocess%20Pipeline.ipynb) relies on `Cython` implementation of [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) project for Go rules and feature plane generation. It is estimated to take 60 CPU hours for preprocess complete KGS data set.

## Supervised Learning - Policy Network
  According to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true), AlphaGo can achieve 55.4% test accuracy after 20 epochs training. Test set is the first 1 million steps. i.e. KGS2004. The speed of each prediction is 4.8ms (on Kepler K40 GPU).  
  [This program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%282%29%20Policy%20Network.ipynb) achieves 54.55% / 52.73% by 8 epochs so far. Test set is the latest 1M steps. i.e.KGS201705-KGS201709. It also achieved speed of around 4.5ms for each single prediction (on Maxwell GTX980m GPU). Therefore each epochs takes ~40 GPU hours. Running on GPU mode is around 100x faster than CPU mode.  
  
| epochs | LR     | loss   | train/test accu | epochs | LR     | loss   | train/test accu |
|--------|--------|--------|-----------------|--------|--------|--------|-----------------|
| 1      | 0.003  | 1.895  | 0.4800 / 0.4724 | 11     |        |        | tbd             |
| 2      | 0.003  | 1.7782 | 0.5118 / 0.4912 | 12     |        |        | tbd             |
| 3      | 0.002  | 1.7110 | 0.5227 / 0.5029 | 13     |        |        | tbd             |
| 4      | 0.002  | 1.6803 | 0.5275 / 0.5079 | 14     |        |        | tbd             |
| 5      | 0.002  | 1.6567 | 0.5312 / 0.5119 | 15     |        |        | tbd             |
| 6      | 0.002  | 1.6376 | 0.5340 / 0.5146 | 16     |        |        | tbd             |
| 7      | 0.001  | 1.6022 | 0.5398 / 0.5202 | 17     |        |        | tbd             |
| 8      | 0.0005 | 1.5782 | 0.5455 / 0.5273 | 18     |        |        | tbd             |
| 9*     | 0.0005 | 1.6039 | 0.5450 / 0.5261 | 19     |        |        | tbd             |
| 10     |        |        | tbd             | 20     |        |        | 0.569/0.554(alphago)|

> epoch 9 may be skipped, the LR does not fit.
> Intel Broadwell CPU can provide around 30 GFlops compute power per core. Nvidia Kepler K40 and Maxwell GTX980m GPU can provide around 3 TFlops compute power.  

## Reinforced Learning - Policy Network
  [The RL program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%283B%29%20Policy%20Network%20-%20Reinforced%20Learning%20in%20mass%20production.ipynb) is runnable now but still under evaluation. It also relies on RocAlphaGo project for Go rules by now. A new program is under construction to implement first 12 features in GPU mode to replace RocAlphaGo. It is believed to be at least 10x faster than RocAlphaGo(python implementation).
  
## Supervised Learning - Value Network
tbd. Depends on Reinforced Learning to generate 30 millions games. And pick 1 state of each game.

## Supervised Learning - Fast Rollout
tbd. AlphaGo achieved 24.2% of accuracy and 2us of speed.

## MTCS
tbd. Depends on Fast Rollout.
