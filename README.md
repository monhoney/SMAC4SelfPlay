# SMAC2Study

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

## 제공되는 실험들
* 여기에서 Max EP Len을 확인하여 학습을 돌리면 된다.
| Name | Ally Units | Enemy Units | Type | Max Ep Len |
| :---: | :---: | :---: | :---:| :---: | 60 | 
| 3m | 3 Marines | 3 Marines | homogeneous & symmetric | 120 |
| 8m | 8 Marines | 8 Marines | homogeneous & symmetric | 150 | 
| 25m | 25 Marines | 25 Marines | homogeneous & symmetric | 120 |
| 2s3z |  2 Stalkers & 3 Zealots |  2 Stalkers & 3 Zealots | heterogeneous & symmetric | 150 |
| 3s5z |  3 Stalkers &  5 Zealots |  3 Stalkers &  5 Zealots | heterogeneous & symmetric | 150 |
| MMM |  1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 2 Marauders & 7 Marines | heterogeneous & symmetric | 70 |
| 5m_vs_6m | 5 Marines | 6 Marines | homogeneous & asymmetric | 120 |
| 8m_vs_9m  | 8 Marines | 9 Marines | homogeneous & asymmetric | 150 |
| 10m_vs_11m | 10 Marines | 11 Marines | homogeneous & asymmetric | 180 |
| 27m_vs_30m | 27 Marines | 30 Marines | homogeneous & asymmetric | 170 |
| 3s5z_vs_3s6z | 3 Stalkers & 5 Zealots | 3 Stalkers & 6 Zealots  | heterogeneous & asymmetric | 180 |
| MMM2 |  1 Medivac, 2 Marauders & 7 Marines |  1 Medivac, 3 Marauders & 8 Marines | heterogeneous & asymmetric | 150 |
| 2m_vs_1z | 2 Marines | 1 Zealot | micro-trick: alternating fire | 300 |
| 2s_vs_1sc| 2 Stalkers  | 1 Spine Crawler | micro-trick: alternating fire | 150 |
| 3s_vs_3z | 3 Stalkers | 3 Zealots | micro-trick: kiting | 150 |
|  3s_vs_4z | 3 Stalkers | 4 Zealots |  micro-trick: kiting | 200 |
| 3s_vs_5z | 3 Stalkers | 5 Zealots |  micro-trick: kiting | 250 |
| 6h_vs_8z | 6 Hydralisks  | 8 Zealots | micro-trick: focus fire | 150 |
| corridor | 6 Zealots  | 24 Zerglings | micro-trick: wall off | 400 |
| bane_vs_bane | 20 Zerglings & 4 Banelings  | 20 Zerglings & 4 Banelings | micro-trick: positioning | 200 |
| so_many_banelings| 7 Zealots  | 32 Banelings | micro-trick: positioning | N/A |
| 2c_vs_64zg| 2 Colossi  | 64 Zerglings | micro-trick: positioning | 400 |
| 1c3s5z | 1 Colossi & 3 Stalkers & 5 Zealots | 1 Colossi & 3 Stalkers & 5 Zealots | heterogeneous & symmetric | 180 |
