# C2TutorialsGo
This is a tutorial written for Caffe2 which mocks google AlphaGo Fan and AlphaGO Zero.

## Preprocess
  The Go game dataset are usually stored in [SGF](http://www.red-bean.com/sgf/go.html) file format. We need to transform SGF file into Caffe2 Tensor which are 48 feature planes of 19x19 size, according to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true).  
    [The preprocess program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%281%29%20Preprocess%20Pipeline.ipynb) relies on `Cython` implementation of [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) project for Go rules and feature plane generation. It is estimated to take 60 CPU hours for preprocess complete KGS data set.

## Supervised Learning - Policy Network
  According to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true), AlphaGo can achieve 55.4% test accuracy after 20 epochs training. Test set is the first 1 million steps. i.e. KGS2004. The speed of each prediction is 4.8ms (on Kepler K40 GPU).  
  [This program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%282%29%20Policy%20Network.ipynb) achieves ~54.6% by ? epochs so far. Test set is the latest 1 million steps. i.e. KGS201705-201709. It also achieved speed of around 4.5ms for each single prediction (on Maxwell GTX980m GPU). Therefore each epochs takes ~40 GPU hours. Running on GPU mode is around 100x faster than CPU mode.  
  
| epochs | LR     | loss   | test accuracy | epochs | LR     | loss   | test accuracy |
|--------|--------|--------|---------------|--------|--------|--------|---------------|
| 1      | 0.003  | 1.895  | 0.4800        | 11     |        |        | tbd           |
| 2      | 0.003  | 1.7782 | 0.5118        | 12     |        |        | tbd           |
| 3      | 0.002  | 1.7110 | 0.5227        | 13     |        |        | tbd           |
| 4      | 0.002  | 1.6803 | 0.5275        | 14     |        |        | tbd           |
| 5      | 0.002  | 1.6567 | 0.5312        | 15     |        |        | tbd           |
| 6      | 0.002  | 1.6376 | 0.5340        | 16     |        |        | tbd           |
| 7      | 0.001  | 1.6022 | 0.5398        | 17     |        |        | tbd           |
| 8      | 0.0005 | 1.5782 | 0.5455/0.5273 | 18     |        |        | tbd           |
| 9      |        |        | tbd           | 19     |        |        | tbd           |
| 10     |        |        | tbd           | 20     |        |        | 0.554(alphago)|

> Intel Broadwell CPU can provide around 30 GFlops compute power per core. Nvidia Kepler K40 and Maxwell GTX980m GPU can provide around 3 TFlops compute power.  

## Reinforced Learning - Policy Network
  [The RL program](http://nbviewer.jupyter.org/github/davinwang/C2TutorialsGo/blob/master/Mock%20AlphaGo%20%283B%29%20Policy%20Network%20-%20Reinforced%20Learning%20in%20mass%20production.ipynb) is runnable now but still under evaluation. It also relies on RocAlphaGo project for Go rules by now. A new program is under construction to implement first 12 features in GPU mode to replace RocAlphaGo. It is believed to be at least 10x faster than RocAlphaGo(python implementation).  
  
## Supervised Learning - Value Network
tbd. Depends on Reinforced Learning to generate 30 millions games. And pick 1 state of each game.

## Supervised Learning - Fast Rollout
tbd. AlphaGo achieved 24.2% of accuracy and 2us of speed.

## MTCS
tbd. Depends on Fast Rollout.

# New updates from AlphaGo Zero
## Dual Policy and Value network with ResNet  
tbd. This will be completed soon. According to DeepMind, the new design will achieve better results for Supervised Learning.
## Reinforced Learning pipline
tbd. This will be different from AlphaGo Fan.
