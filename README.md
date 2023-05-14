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
