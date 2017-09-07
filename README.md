# C2TutorialsGo
This is a tutorial written for Caffe2 which mocks google AlphaGo.

## Preprocess
This program relies on RocAlphaGo project for Go rules and feature plane generation.
Since it is implemented in python instead of C/C++, the preprocess takes around 500 CPU hours for complete KGS data set.

## Supervised Learning - Policy Network
According to DeepMind, AlphaGo can achieve 55.4% accuracy after 20 epochs training. Test set is the first 1 million steps. i.e. KGS2004. The speed of each prediction is 4.8ms (on Kepler GPU).
This program achieves 52.75% accuracy by 4 epochs so far. Test set is the latest 300K steps. i.e. KGS201705-201706. It also achieved speed of around 4.5ms for each single prediction (on Maxwell GPU).
Each epochs takes ~40 GPU hours(on Maxwell GPU). Running on GPU mode is around 100x faster than CPU mode.

## Reinforced Learning - Policy Network
The program is runnable but still under evaluation. It also relies on RocAlphaGo project for Go rules by now.
A new program is under construction to implement first 12 features in GPU mode to replace RocAlphaGo. It is believed to be at least 10x faster than RocAlphaGo.

## Supervised Learning - Value Network
tbd. Depends on Reinforced Learning to generate 30 millions games. And pick 1 state of each game.

## Supervised Learning - Fast Rollout
tbd. AlphaGo achieved 24.2% of accuracy and 2us of speed.

## MTCS
tbd. Depends on Fast Rollout.
