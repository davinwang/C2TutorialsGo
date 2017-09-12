# C2TutorialsGo
This is a tutorial written for Caffe2 which mocks google AlphaGo.

## Preprocess
  The Go game dataset are usually stored in [SGF](http://www.red-bean.com/sgf/go.html) file format. We need to transform SGF file into Caffe2 Tensor which are 48 feature planes of 19x19 size, according to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true).  
    The preprocess program relies on `Cython` implementation of [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) project for Go rules and feature plane generation. It is estimated to take 60 CPU hours for preprocess complete KGS data set.

## Supervised Learning - Policy Network
  According to [DeepMind](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true), AlphaGo (conv=13 filters=192) can achieve 55.4% test accuracy after 20 epochs training. Test set is the first 1 million steps. i.e. KGS2004. The speed of each prediction is 4.8ms (on Kepler GPU).  
  This program (conv=13 filters=192) achieves 52.75% test accuracy by 4 epochs and 54% by 7 epochs so far. Test set is the latest 300K steps. i.e. KGS201705-201706. It also achieved speed of around 4.5ms for each single prediction (on Maxwell GPU). Therefore each epochs takes ~40 GPU hours. Running on GPU mode is around 100x faster than CPU mode.  
  
| epochs | loss   | test accuracy | epochs | loss   | test accuracy |
|--------|--------|---------------|--------|--------|---------------|
| 1      | 1.895  | 0.4800        | 11     |        | tbd           |
| 2      | 1.7782 | 0.5118        | 12     |        | tbd           |
| 3      | 1.7110 | 0.5227        | 13     |        | tbd           |
| 4      | 1.6803 | 0.5275        | 14     |        | tbd           |
| 5      | 1.6567 | 0.5312        | 15     |        | tbd           |
| 6      | 1.6376 | 0.5340        | 16     |        | tbd           |
| 7      | 1.6022 | 0.5398        | 17     |        | tbd           |
| 8      |        | tbd           | 18     |        | tbd           |
| 9      |        | tbd           | 19     |        | tbd           |
| 10     |        | tbd           | 20     |        | 0.554(alphago)|

> Intel Broadwell CPU can provide around 30 GFlops compute power per core. Nvidia Kepler K40 and Maxwell GTX980m GPU can provide around 3 TFlops compute power.  

## Reinforced Learning - Policy Network
  The program is runnable but still under evaluation. It also relies on RocAlphaGo project for Go rules by now. A new program is under construction to implement first 12 features in GPU mode to replace RocAlphaGo. It is believed to be at least 10x faster than RocAlphaGo(python implementation).  
  
| Black Player <br> conv/filters/features/SL/RL | White Player <br> conv/filters/features/SL/RL | compete result <br> black : white |  
|-----------------------------------|-------------------------------|--------|
| **13 / 192 / 48 /  1 epoch /  0** | 13 / 192 / 48 /  1 epoch /  0 |  9 : 7 |
| **13 / 192 / 48 /  4 epoch /  0** | 13 / 192 / 48 /  1 epoch /  0 |  9 : 7 |
| 13 / 192 / 48 /  1 epoch /  0 | **13 / 192 / 48 /  4 epoch /  0** |  4 : 12 |
| 13 / 192 / 48 /  4 epoch /  0 | **13 / 192 / 48 /  4 epoch /  0** |  5 : 11 |
| **13 / 192 / 48 /  6 epoch /  0** | 13 / 192 / 48 /  1 epoch /  0 |  13 : 3 |
| 13 / 192 / 48 /  1 epoch /  0 | **13 / 192 / 48 /  6 epoch /  0** |  4 : 12 |
| 13 / 192 / 48 /  6 epoch /  0 | **13 / 192 / 48 /  4 epoch /  0** |  7 : 9 |
| 13 / 192 / 48 /  4 epoch /  0 | 13 / 192 / 48 /  6 epoch /  0 |  8 : 8 |
| 13 / 192 / 48 /  6 epoch /  0 | **13 / 192 / 48 /  6 epoch /  0** |  6 : 10 |
  
## Supervised Learning - Value Network
tbd. Depends on Reinforced Learning to generate 30 millions games. And pick 1 state of each game.

## Supervised Learning - Fast Rollout
tbd. AlphaGo achieved 24.2% of accuracy and 2us of speed.

## MTCS
tbd. Depends on Fast Rollout.
