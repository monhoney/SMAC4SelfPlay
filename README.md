# SMAC4SelfPlay

# 설치

리눅스를 기준으로 설치방법을 소개하겠음.

## StarCraft2 설치
* https://github.com/oxwhirl/smac 를 참고해서 설치하면 된다. 

보기 귀찮은 경우에...
* StarCraft2 리눅스 배포버전 중에 가장 최신 버전을 설치한다.
	- 설치는 $HOME/StarCraftII에 하는 것을 추천한다. 그렇지 않을 경우에는 환경변수 SC2PATH를 설정해주어야 한다.
	- 압축을 풀때 암호는 iagreetotheeula 이다.
```
$ cd $HOME
$ wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
$ unzip SC2.4.10.zip
```

SMAC_maps를 설치한다.
```
$ cd $HOME/StarCraft2/Maps
$ wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
$ unzip SMAC_Maps.zip
```

## conda환경 및 pip 및 SMAC 설치
```
$ conda create -n SMAC python=3
$ pip install -r requirements.txt
$ cd 3rd_library/smac
$ pip install -e .
```

## 실행방법
```
$ python algo/mppo/mppo.py --env 8m --eplen 120 --epochs 1000
```

## 결과물 예시

### 3m
![](https://github.com/monhoney/SMAC2Study/blob/main/asset/3m.gif)

### 8m
![](https://github.com/monhoney/SMAC2Study/blob/main/asset/8m.gif)

## 제공되는 실험들
| Name | Ally Units | Enemy Units | Type | Max Ep Len | 
| :---: | :---: | :---: | :---:| :---: |
| 3m | 3 Marines | 3 Marines | homogeneous & symmetric | 60 |
| 8m | 8 Marines | 8 Marines | homogeneous & symmetric | 120 | 
| 25m | 25 Marines | 25 Marines | homogeneous & symmetric | 150 |
| 2s3z |  2 Stalkers & 3 Zealots |  2 Stalkers & 3 Zealots | heterogeneous & symmetric | 120 |
| 3s5z |  3 Stalkers &  5 Zealots |  3 Stalkers &  5 Zealots | heterogeneous & symmetric | 150 |
| MMM |  1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 2 Marauders & 7 Marines | heterogeneous & symmetric | 150 |
| 5m_vs_6m | 5 Marines | 6 Marines | homogeneous & asymmetric | 70 |
| 8m_vs_9m  | 8 Marines | 9 Marines | homogeneous & asymmetric | 120 |
| 10m_vs_11m | 10 Marines | 11 Marines | homogeneous & asymmetric | 150 |
| 27m_vs_30m | 27 Marines | 30 Marines | homogeneous & asymmetric | 180 |
| 3s5z_vs_3s6z | 3 Stalkers & 5 Zealots | 3 Stalkers & 6 Zealots  | heterogeneous & asymmetric | 170 |
| MMM2 |  1 Medivac, 2 Marauders & 7 Marines |  1 Medivac, 3 Marauders & 8 Marines | heterogeneous & asymmetric | 180 |
| 2m_vs_1z | 2 Marines | 1 Zealot | micro-trick: alternating fire | 150 |
| 2s_vs_1sc| 2 Stalkers  | 1 Spine Crawler | micro-trick: alternating fire | 300 |
| 3s_vs_3z | 3 Stalkers | 3 Zealots | micro-trick: kiting | 150 |
|  3s_vs_4z | 3 Stalkers | 4 Zealots |  micro-trick: kiting | 200 |
| 3s_vs_5z | 3 Stalkers | 5 Zealots |  micro-trick: kiting | 250 |
| 6h_vs_8z | 6 Hydralisks  | 8 Zealots | micro-trick: focus fire | 150 |
| corridor | 6 Zealots  | 24 Zerglings | micro-trick: wall off | 400 |
| bane_vs_bane | 20 Zerglings & 4 Banelings  | 20 Zerglings & 4 Banelings | micro-trick: positioning | 200 |
| so_many_banelings| 7 Zealots  | 32 Banelings | micro-trick: positioning | N/A |
| 2c_vs_64zg| 2 Colossi  | 64 Zerglings | micro-trick: positioning | 400 |
| 1c3s5z | 1 Colossi & 3 Stalkers & 5 Zealots | 1 Colossi & 3 Stalkers & 5 Zealots | heterogeneous & symmetric | 180 |

# SMAC for Self-Play

## 준비 사항
* SelfPlay용 맵이 따로 존재하며 현재는 1개의 맵(`Simple64_Tank.SC2Map`)를 지원되고 있다. 
* 참고로 해당 맵은 `3rd_library/smac/smac/env/starcraft2/maps/SMAC_Maps` 폴더 아래에 있다.
* 맵 파일을 스타크래프트가 설치된 폴더(ex. `$HOME/StarCraftII/Maps/SMAC_Maps`)에 복사를 하면 된다.
```
$ cp $BASE_DIR/3rd_library/smac/smac/env/starcraft2/maps/SMAC_Maps/Simple64_Tank.SC2Map $HOME/StarCraftII/Maps/SMAC_Maps/.
```

## 학습방법
실험명을 `TEST_EXP_NAME`이라고 하는 경우에 아래와 같은 형태로 학습을 하면 된다.
```
$ python algo/mppo/mppo_sp.py --env Simple64_Tank --eplen 80 --epochs 1000 --exp_name TEST_EXP_NAME
```

## 학습 결과 확인 방법
* 실험명이 `TEST_EXP_NAME`인 경우 기본적으로 모델 파일은 `$BASE_DIR/TEST_EXP_NAME/TEST_EXP_NAME_s0/pyt_save/model.pt`라는 이름으로 저장이 된다.
* 아래와 같이 학습 결과물을 5회 테스트를 할 수 있다.
```
$ python algo/mppo/mppo_sp.py --test --model_filepath $BASE_DIR/TEST_EXP_NAME/TEST_EXP_NAME_s0/pyt_save/model.pt
```

# SMAC for Self-Play version2

## 설명
* 기존 SMAC for Self-Play의 경우에 PPO와 Framework가 tightly coupled 되어 있어서 새로운 알고리즘을 돌리기 어려웠음
* 현재 SMAC for Self-Play ver2의 경우에 알고리즘을 등록하면 사용할 수 있는 구조로 되어 있음

## 알고리즘 추가 방법
* `algo2` 폴더 아래에 알고리즘이름(ex. ABC)으로 폴더를 만들고 그 밑에 `run.py` 파일을 만든다.
* `run.py`파일은 `algo2/wrapper.py`의 `RLAlgoWrapper`를 상속한 ABC 클래스로 작성한다. `RLAlgoWrapper`에서 사용된 함수들은 모두 구현해야 한다.
* `algo2/register.py`파일에 추가할 ABC 알고리즘에 대한 정보를 등록한다.

## 학습방법
* player이 학습이 될 알고리즘이고, enemy가 상대편 알고리즘이다.
* player를 PPO, enemy를 RANDOM으로 학습하는 경우에 아래와 같이 학습을 한다.
```
$ python train.py --player_algo PPO --enemy_algo RANDOM 
```

### 학습한 모델끼리 학습을 하는 경우
* `data/2023-10-16_selfplay/2023-10-16_21-23-30-selfplay_s0/pyt_save` 에 저장된 PPO 모델끼리 학습하는 경우에 아래와 같이 학습을 한다.
```
$ python train.py --player_algo PPO --player_model_path data/2023-10-16_selfplay/2023-10-16_21-23-30-selfplay_s0/pyt_save'--enemy_algo PPO --enemy_model_path data/2023-10-16_selfplay/2023-10-16_21-23-30-selfplay_s0/pyt_save'--enemy_algo
```

